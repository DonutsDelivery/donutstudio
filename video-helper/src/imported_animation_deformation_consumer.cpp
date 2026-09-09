#include "imported_animation_deformation_consumer.h"

#include "sha256.h"
#include "../../shared/Visual3DScene.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <utility>
#include <vector>

namespace videohelper
{
namespace
{
bool boundedText (const std::string& value, std::size_t maximum) noexcept
{
    return ! value.empty() && value.size() <= maximum
        && std::none_of(value.begin(), value.end(), [] (unsigned char character)
        { return character < 0x20 || character == 0x7f; });
}

bool stableAssetId (const std::string& value) noexcept
{
    if (! boundedText(value, 128)) return false;
    return std::all_of(value.begin(), value.end(), [] (unsigned char character)
    {
        return std::isalnum(character) != 0 || character == '-'
            || character == '_' || character == '.';
    });
}

bool lowercaseSha256 (const std::string& value) noexcept
{
    return value.size() == 64
        && std::all_of(value.begin(), value.end(), [] (unsigned char character)
        { return std::isdigit(character) != 0 || (character >= 'a' && character <= 'f'); });
}

bool validateRequestIdentity (const visualanimationimport::Request& request,
                              std::string& error)
{
    if (request.sourceStableId == 0 || request.deformationStableId == 0
        || request.schedule != std::array<visualanimationimport::StableId, 2> {
               request.sourceStableId, request.deformationStableId })
    {
        error = "imported animation/deformation schedule has invalid stable identities";
        return false;
    }
    if (! stableAssetId(request.asset.id) || request.asset.version == 0
        || request.asset.version > static_cast<std::uint64_t>(
               std::numeric_limits<std::int64_t>::max())
        || ! lowercaseSha256(request.asset.contentSha256)
        || (request.clipName.empty() ? request.animationClipStableId == 0
                                     : ! boundedText(request.clipName, 256))
        || (request.combinationMode != visualanimation::CombinationMode::Replace
            && request.combinationMode != visualanimation::CombinationMode::WeightedBlend
            && request.combinationMode != visualanimation::CombinationMode::Add
            && request.combinationMode != visualanimation::CombinationMode::Multiply
            && request.combinationMode
                != visualanimation::CombinationMode::AddAfterImportedAnimation))
    {
        error = "imported animation/deformation request identity is malformed";
        return false;
    }
    if (request.asset.sourceMediaType != "model/gltf-binary")
    {
        error = "imported animation/deformation resource type is unsupported";
        return false;
    }
    if (request.asset.sourceByteSize == 0
        || request.asset.sourceByteSize > 512ull * 1024ull * 1024ull)
    {
        error = "imported animation/deformation exact-content size is invalid";
        return false;
    }
    return true;
}

const gltf::GlbNamedAnimationClip* findNamedClip (
    const gltf::GlbAnimationDocument& document,
    const std::string& name,
    std::string& error)
{
    const gltf::GlbNamedAnimationClip* result = nullptr;
    for (const auto& clip : document.clips)
    {
        if (clip.name != name) continue;
        if (result != nullptr)
        {
            error = "imported animation/deformation named clip is ambiguous";
            return nullptr;
        }
        result = &clip;
    }
    if (result == nullptr)
        error = "imported animation/deformation named clip is unavailable";
    return result;
}

const gltf::GlbNamedAnimationClip* findSelectedClip (
    const gltf::GlbAnimationDocument& document,
    const visualanimationimport::Request& request,
    std::string& error)
{
    if (request.animationClipStableId == 0)
        return findNamedClip(document, request.clipName, error);

    const auto index = request.animationClipStableId - 1;
    if (index >= document.clips.size())
    {
        error = "imported animation/deformation clip selector is unavailable";
        return nullptr;
    }
    const auto& selected = document.clips[static_cast<std::size_t>(index)];
    if (selected.name != request.clipName)
    {
        error = "imported animation/deformation clip selector and name disagree";
        return nullptr;
    }
    return &selected;
}

bool hasSelectedMesh(const gltf::GlbAnimationDocument& document,
                     const visualanimationimport::Request& request,
                     std::string& error)
{
    if (request.meshStableId == 0) return true;
    const auto found = std::any_of(
        document.renderBindings.begin(), document.renderBindings.end(),
        [&request] (const auto& binding)
        { return binding.mesh.value == request.meshStableId; });
    if (!found)
        error = "imported animation/deformation mesh selector is unavailable";
    return found;
}

bool selectedClipDrivesMesh(const gltf::GlbAnimationDocument& document,
                            const gltf::GlbNamedAnimationClip& clip,
                            const visualanimationimport::Request& request,
                            std::string& error)
{
    if (request.meshStableId == 0) return true;
    const auto compatible = importedAnimationCompatibleMeshStableIds(document, clip);
    const auto driven = std::find(compatible.begin(), compatible.end(), request.meshStableId)
        != compatible.end();
    if (!driven)
        error = "imported animation/deformation clip does not drive selected mesh";
    return driven;
}
} // namespace

gltf::GlbAnimationDecodeOptions nativeImportedAnimationDecodeOptions(
    std::size_t size, std::optional<std::size_t> sceneIndex)
{
    gltf::GlbAnimationDecodeOptions options;
    options.admission.admitAnimations = true;
    options.admission.admitSkins = true;
    options.admission.limits.maxContainerBytes
        = std::min(options.admission.limits.maxContainerBytes, size);
    options.admission.limits.maxTextures = HarmonicMIDI::grid::Visual3DScene::kMaxTextures;
    options.admission.limits.maxImages = HarmonicMIDI::grid::Visual3DScene::kMaxTextures;
    options.admission.limits.maxSamplers = HarmonicMIDI::grid::Visual3DScene::kMaxTextures;
    options.admission.limits.maxEmbeddedImageBytes
        = HarmonicMIDI::grid::Visual3DScene::kMaxTextureTexels * 4u;
    options.admission.limits.maxImageWidth = 8192;
    options.admission.limits.maxImageHeight = 8192;
    options.admission.limits.maxDecodedImageBytes
        = HarmonicMIDI::grid::Visual3DScene::kMaxTextureTexels * 4u;
    options.admission.supportedRequiredExtensions = {"KHR_lights_punctual"};
    options.admission.sceneIndex = sceneIndex;
    return options;
}

std::vector<visualanimationimport::StableId> importedAnimationCompatibleMeshStableIds(
    const gltf::GlbAnimationDocument& document,
    const gltf::GlbNamedAnimationClip& clip)
{
    std::vector<visualanimationimport::StableId> compatible;
    compatible.reserve(document.renderBindings.size());
    for (const auto& renderBinding : document.renderBindings)
    {
        const auto morphDriven = std::any_of(
            clip.morphBindings.begin(), clip.morphBindings.end(),
            [&renderBinding] (const auto& binding)
            { return binding.mesh == renderBinding.mesh; });
        const auto jointDriven = std::any_of(
            clip.jointBindings.begin(), clip.jointBindings.end(),
            [&renderBinding] (const auto& animated)
            {
                return std::any_of(
                    renderBinding.jointBaseTransforms.begin(),
                    renderBinding.jointBaseTransforms.end(),
                    [&animated] (const auto& base) { return base.skin == animated.skin; });
            });
        if ((morphDriven || jointDriven)
            && std::find(compatible.begin(), compatible.end(), renderBinding.mesh.value)
                == compatible.end())
            compatible.push_back(renderBinding.mesh.value);
    }
    return compatible;
}

bool ImportedAnimationDeformationConsumer::admit (
    const visualanimationimport::Request& request,
    const std::uint8_t* bytes,
    std::size_t size,
    std::string& error)
{
    error.clear();
    if (! validateRequestIdentity(request, error)) return false;
    if (bytes == nullptr || size != request.asset.sourceByteSize)
    {
        error = "imported animation/deformation bytes do not match exact-content size";
        return false;
    }

    Sha256 hash;
    hash.update(bytes, size);
    if (hash.finishHex() != request.asset.contentSha256)
    {
        error = "imported animation/deformation bytes do not match exact-content fingerprint";
        return false;
    }

    auto options = nativeImportedAnimationDecodeOptions(size, request.sceneIndex);
    auto decoded = gltf::decodeGlbAnimations(bytes, size, options, error);
    if (! decoded)
    {
        error = "imported animation/deformation decode rejected: " + error;
        return false;
    }
    const auto* selectedClip = findSelectedClip(*decoded, request, error);
    if (selectedClip == nullptr) return false;
    if (!hasSelectedMesh(*decoded, request, error)) return false;
    if (!selectedClipDrivesMesh(*decoded, *selectedClip, request, error)) return false;

    auto admitted = std::make_shared<gltf::GlbAnimationDocument>(std::move(*decoded));
    asset_ = request.asset;
    document_ = std::move(admitted);
    error.clear();
    return true;
}

bool ImportedAnimationDeformationConsumer::evaluatePreview (
    const visualanimationimport::Request& request,
    const visualdeformation::RationalFrameTime& frame,
    std::uint64_t structuralRevision,
    ImportedAnimationDeformationEvaluation& destination,
    std::string& error) const
{
    return evaluate(ImportedAnimationEvaluationOwner::Preview, request, frame,
                    structuralRevision, destination, error);
}

bool ImportedAnimationDeformationConsumer::evaluateExport (
    const visualanimationimport::Request& request,
    const visualdeformation::RationalFrameTime& frame,
    std::uint64_t structuralRevision,
    ImportedAnimationDeformationEvaluation& destination,
    std::string& error) const
{
    return evaluate(ImportedAnimationEvaluationOwner::Export, request, frame,
                    structuralRevision, destination, error);
}

bool ImportedAnimationDeformationConsumer::evaluate (
    ImportedAnimationEvaluationOwner owner,
    const visualanimationimport::Request& request,
    const visualdeformation::RationalFrameTime& frame,
    std::uint64_t structuralRevision,
    ImportedAnimationDeformationEvaluation& destination,
    std::string& error) const
{
    error.clear();
    if (! validateRequestIdentity(request, error)) return false;
    if (! document_ || ! visualanimationimport::sameAsset(asset_, request.asset))
    {
        error = "imported animation/deformation exact content is not admitted";
        return false;
    }
    if (structuralRevision == 0)
    {
        error = "imported animation/deformation structural revision is missing";
        return false;
    }

    const auto* named = findSelectedClip(*document_, request, error);
    if (named == nullptr || ! named->clip || ! document_->deformation) return false;

    const auto playback = visualanimation::resolvePlaybackControl(
        *named->clip, request.playback, {}, error);
    if (! playback) return false;

    std::vector<visualdeformation::JointAnimationBindingView> joints;
    joints.reserve(named->jointBindings.size());
    for (const auto& binding : named->jointBindings)
        joints.push_back({binding.animationTarget, binding.skin, binding.joint});
    std::vector<visualdeformation::MorphAnimationBindingView> morphs;
    morphs.reserve(named->morphBindings.size());
    for (const auto& binding : named->morphBindings)
        morphs.push_back({binding.animationTarget, binding.mesh,
                          binding.targets.data(), binding.targets.size()});

    visualdeformation::AnimationDeformationRequest evaluationRequest;
    evaluationRequest.time = frame;
    evaluationRequest.revision = structuralRevision;
    evaluationRequest.playback = playback->sample.playback;
    evaluationRequest.requestedTimeSeconds = playback->sample.timeSeconds;
    evaluationRequest.sampleRangeStartSeconds = playback->sample.rangeStartSeconds;
    evaluationRequest.sampleRangeEndSeconds = playback->sample.rangeEndSeconds;
    evaluationRequest.combinationMode = request.combinationMode;
    evaluationRequest.combinationWeight = playback->weight;
    const visualdeformation::AnimationDeformationBindingView bindings {
        joints.data(), joints.size(), morphs.data(), morphs.size()
    };
    auto deformation = visualdeformation::evaluateAnimationDeformation(
        *document_->deformation, *named->clip, bindings, evaluationRequest, {}, error);
    if (! deformation) return false;

    ImportedAnimationDeformationEvaluation candidate;
    candidate.owner = owner;
    candidate.sourceStableId = request.sourceStableId;
    candidate.deformationStableId = request.deformationStableId;
    candidate.playback = *playback;
    candidate.deformation = std::move(deformation);
    destination = std::move(candidate);
    error.clear();
    return true;
}
} // namespace videohelper
