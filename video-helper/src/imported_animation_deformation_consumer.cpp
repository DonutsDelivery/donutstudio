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
    if (!visualanimation::validPoseControls(request.pose)
        || ((request.pose.boneEnabled || request.pose.morphEnabled)
            && request.pose.meshStableId != request.meshStableId
            && !(request.meshStableId == 0 && request.pose.nodeStableId != 0)))
    {
        error = "imported animation controls are malformed or out of bounds";
        return false;
    }
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
    const auto found = std::any_of(document.nodeMeshes.begin(), document.nodeMeshes.end(),
        [&request] (const auto& mesh) { return mesh.value == request.meshStableId; });
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

bool validateSelectedPose(const gltf::GlbAnimationDocument& document,
                          const visualanimationimport::Request& request,
                          std::string& error,
                          visualanimation::PoseControls* resolved = nullptr)
{
    const auto& pose = request.pose;
    if (resolved != nullptr) *resolved = pose;
    if (pose.nodeStableId != 0 && (pose.nodeStableId > document.nodeMeshes.size()
        || document.nodeMeshes[pose.nodeStableId - 1u].value != pose.meshStableId
        || (request.meshStableId != 0 && request.meshStableId != pose.meshStableId)))
    { error = "selected pose object does not belong to the imported mesh selection"; return false; }
    if (!pose.boneEnabled && !pose.morphEnabled) return true;
    const auto* mesh = document.deformation
        ? document.deformation->findMesh({pose.meshStableId}) : nullptr;
    if (mesh == nullptr || (request.meshStableId != 0 && pose.meshStableId != request.meshStableId))
    {
        error = "selected pose controls do not belong to the imported mesh";
        return false;
    }
    const gltf::GlbDeformationRenderBinding* selected = nullptr;
    for (const auto& binding : document.renderBindings)
        if (binding.mesh == mesh->id()
            && (pose.nodeStableId == 0 || binding.nodeIndex + 1u == pose.nodeStableId))
        {
            if (selected != nullptr)
            { error = "selected pose mesh has multiple objects; select an exact object node ID"; return false; }
            selected = &binding;
        }
    if (!selected) { error = "selected pose object has no deformation binding"; return false; }
    if (resolved != nullptr) resolved->nodeStableId = selected->nodeIndex + 1u;
    const auto* skin = document.deformation->findSkin(selected->skin);
    if (pose.boneEnabled && (skin == nullptr
        || std::none_of(skin->joints().begin(),
            skin->joints().begin() + std::min(selected->authoredJointCount, skin->joints().size()),
            [&pose] (const auto& joint) { return joint.id().value == pose.boneStableId; })))
    {
        error = "selected bone does not belong to the imported mesh skin";
        return false;
    }
    const bool unresolved = pose.morphTargetIndex == visualanimation::kUnresolvedMorphTargetIndex;
    if (pose.morphEnabled && !unresolved && pose.morphTargetIndex >= mesh->morphTargets().size())
    {
        error = "selected morph target does not belong to the imported mesh";
        return false;
    }
    if (pose.morphEnabled)
    {
        std::size_t firstCatalogId = 1u;
        for (const auto& preceding : document.deformation->meshes())
            if (preceding.id().value < mesh->id().value)
                firstCatalogId += preceding.morphTargets().size();
        if (pose.morphTargetStableId < firstCatalogId
            || pose.morphTargetStableId - firstCatalogId >= mesh->morphTargets().size()
            || (!unresolved && firstCatalogId + pose.morphTargetIndex != pose.morphTargetStableId))
        {
            error = "selected morph catalog identity and mesh target disagree";
            return false;
        }
        if (resolved != nullptr)
            resolved->morphTargetIndex = static_cast<std::uint32_t>(pose.morphTargetStableId - firstCatalogId);
    }
    return true;
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
    options.admission.limits.maxImageWidth
        = HarmonicMIDI::grid::Visual3DScene::kMaxTextureDimension;
    options.admission.limits.maxImageHeight
        = HarmonicMIDI::grid::Visual3DScene::kMaxTextureDimension;
    options.admission.limits.maxDecodedImageBytes
        = HarmonicMIDI::grid::Visual3DScene::kMaxTextureTexels * 4u;
    options.admission.supportedRequiredExtensions = {"KHR_lights_punctual"};
    options.admission.sceneIndex = sceneIndex;
    options.retainGeometryHierarchy = true;
    return options;
}

std::vector<visualanimationimport::StableId> importedAnimationCompatibleMeshStableIds(
    const gltf::GlbAnimationDocument& document,
    const gltf::GlbNamedAnimationClip& clip)
{
    std::vector<visualanimationimport::StableId> compatible;
    if (!clip.sceneTransformTargets.empty())
        for (const auto mesh : document.sceneMeshes) compatible.push_back(mesh.value);
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
    std::string& error, bool geometryExtraction)
{
    error.clear();
    if (! validateRequestIdentity(request, error)) return false;
    if (geometryExtraction && request.meshStableId == 0)
    { error = "Animated Geometry3D requires one selected mesh"; return false; }
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
    if (request.meshStableId != 0)
    {
        if (!hasSelectedMesh(*decoded, request, error)) return false;
        for (auto& named : decoded->clips)
        {
            named.jointBindings.erase(std::remove_if(named.jointBindings.begin(),named.jointBindings.end(),
                [&](const auto& binding) {
                    return std::none_of(decoded->renderBindings.begin(), decoded->renderBindings.end(),
                        [&](const auto& draw) { return draw.mesh.value == request.meshStableId && draw.skin == binding.skin; });
                }),named.jointBindings.end());
            named.morphBindings.erase(std::remove_if(named.morphBindings.begin(),named.morphBindings.end(),
                [&](const auto& binding) { return binding.mesh.value != request.meshStableId; }),named.morphBindings.end());
            std::vector<visualanimation::TrackView> tracks;
            for (const auto& track : named.clip->tracks())
            {
                const bool selected = track.channel()==visualanimation::Channel::MorphWeights
                    ? std::any_of(named.morphBindings.begin(),named.morphBindings.end(),[&](const auto& b) { return b.animationTarget==track.target(); })
                    : std::any_of(named.jointBindings.begin(),named.jointBindings.end(),[&](const auto& b) { return b.animationTarget==track.target(); });
                const bool sceneTransform = track.channel()!=visualanimation::Channel::MorphWeights
                    && std::find(named.sceneTransformTargets.begin(),named.sceneTransformTargets.end(),track.target())
                        != named.sceneTransformTargets.end();
                if (selected || sceneTransform) tracks.push_back({track.id(),track.target(),track.channel(),track.interpolation(),
                    track.keyTimesSeconds().data(),track.values().data(),track.keyCount(),track.values().size(),
                    track.channel()==visualanimation::Channel::MorphWeights ? track.valueWidth() : 0});
            }
            if (tracks.empty()) continue;
            const visualanimation::ClipView view{named.clip->id(),named.clip->durationSeconds(),tracks.data(),tracks.size()};
            auto filtered=visualanimation::Clip::create(view,{},error);
            if (!filtered) return false;
            named.clip=std::move(filtered);
        }
    }
    const auto* selectedClip = findSelectedClip(*decoded, request, error);
    if (selectedClip == nullptr) return false;
    if (!hasSelectedMesh(*decoded, request, error)) return false;
    if (!selectedClipDrivesMesh(*decoded, *selectedClip, request, error)) return false;
    if (!validateSelectedPose(*decoded, request, error)) return false;

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
    if (frame.rateNumerator==0 || frame.rateDenominator==0)
    { error="imported animation frame rate is missing"; return false; }

    const auto* named = findSelectedClip(*document_, request, error);
    if (named == nullptr || !named->clip) return false;
    visualanimation::PoseControls resolvedPose;
    if (!validateSelectedPose(*document_, request, error, &resolvedPose)) return false;

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
    evaluationRequest.pose = resolvedPose;
    evaluationRequest.allowNodeTransformAndMorph = true;
    const visualdeformation::AnimationDeformationBindingView bindings {
        joints.data(), joints.size(), morphs.data(), morphs.size(),
        named->sceneTransformTargets.data(), named->sceneTransformTargets.size()
    };
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation;
    if (document_->deformation)
    {
        deformation = visualdeformation::evaluateAnimationDeformation(
            *document_->deformation, *named->clip, bindings, evaluationRequest, {}, error);
        if (!deformation) return false;
    }
    std::shared_ptr<const visualanimation::Sample> sceneAnimation;
    if (!named->sceneTransformTargets.empty())
    {
        auto sampled = visualanimation::sample(*named->clip, playback->sample, error);
        if (!sampled) return false;
        sceneAnimation = std::make_shared<const visualanimation::Sample>(std::move(*sampled));
    }
    if (!deformation && !sceneAnimation)
    { error = "imported animation has no supported mesh, camera or light targets"; return false; }

    ImportedAnimationDeformationEvaluation candidate;
    candidate.owner = owner;
    candidate.sourceStableId = request.sourceStableId;
    candidate.deformationStableId = request.deformationStableId;
    candidate.playback = *playback;
    candidate.deformation = std::move(deformation);
    candidate.sceneAnimation = std::move(sceneAnimation);
    destination = std::move(candidate);
    error.clear();
    return true;
}
} // namespace videohelper
