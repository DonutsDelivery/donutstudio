#pragma once

#include "../../shared/VisualAnimationDeformationEvaluation.h"
#include "../../shared/VisualAnimationPlaybackControl.h"
#include "../../shared/VisualImportedAnimationDeformationRequest.h"
#include "gltf_glb.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace videohelper
{
std::vector<visualanimationimport::StableId> importedAnimationCompatibleMeshStableIds(
    const gltf::GlbAnimationDocument& document,
    const gltf::GlbNamedAnimationClip& clip);

gltf::GlbAnimationDecodeOptions nativeImportedAnimationDecodeOptions(
    std::size_t size,
    std::optional<std::size_t> sceneIndex = std::nullopt);

enum class ImportedAnimationEvaluationOwner
{
    Preview,
    Export
};

struct ImportedAnimationDeformationEvaluation final
{
    ImportedAnimationEvaluationOwner owner = ImportedAnimationEvaluationOwner::Preview;
    visualanimationimport::StableId sourceStableId = 0;
    visualanimationimport::StableId deformationStableId = 0;
    visualanimation::ResolvedPlaybackControl playback;
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation;
};

// Production admission/evaluation seam for graph-lowered imported-animation
// requests. Admission accepts caller-owned bytes only, verifies their exact
// identity, and decodes once. Preview and export evaluations share the same
// deterministic implementation and never acquire content from a path.
class ImportedAnimationDeformationConsumer final
{
public:
    bool admit (const visualanimationimport::Request& request,
                const std::uint8_t* bytes,
                std::size_t size,
                std::string& error);

    bool evaluatePreview (const visualanimationimport::Request& request,
                          const visualdeformation::RationalFrameTime& frame,
                          std::uint64_t structuralRevision,
                          ImportedAnimationDeformationEvaluation& destination,
                          std::string& error) const;

    bool evaluateExport (const visualanimationimport::Request& request,
                         const visualdeformation::RationalFrameTime& frame,
                         std::uint64_t structuralRevision,
                         ImportedAnimationDeformationEvaluation& destination,
                         std::string& error) const;

    std::shared_ptr<const gltf::GlbAnimationDocument> admittedDocument() const noexcept
    {
        return document_;
    }

private:
    bool evaluate (ImportedAnimationEvaluationOwner owner,
                   const visualanimationimport::Request& request,
                   const visualdeformation::RationalFrameTime& frame,
                   std::uint64_t structuralRevision,
                   ImportedAnimationDeformationEvaluation& destination,
                   std::string& error) const;

    visualanimationimport::ExactContentAssetKey asset_;
    std::shared_ptr<const gltf::GlbAnimationDocument> document_;
};
} // namespace videohelper
