#pragma once

#include "native_scene_animation_composition.h"
#include "native_animation_deformation_renderer.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

namespace videohelper::modelpayload
{
// Logical CPU retention only. Allocator, library and native-driver overhead is
// unknown. The pool uses four history envelopes plus one attempt envelope.
// Actual originals, caller scratch and escaped references all consume this pool;
// it refuses rather than promise four maximum-sized original-input admissions.
class NativeCompositionRetentionBudget final
{
public:
    NativeCompositionRetentionBudget() : bytes_(sizeof(*this)) {}
    static constexpr std::size_t kCallerMaximumBytes = 64 *
        (3 * sizeof(arbitgpu::NativeDeformationScene)
         + 3 * 64 * sizeof(arbitgpu::NativeDeformationJointBaseTransform)
         + 3 * 8 * sizeof(float) + 4 * sizeof(std::shared_ptr<const arbitgpu::NativeDeformationScene>)
         + sizeof(arbitgpu::NativeDeformationCompositionMapping));
    static constexpr std::size_t kMaximumBytes = 5 *
        (arbitgpu::NativeDeformationCompositionOwner::kMaximumLogicalBytes * 3 / 2
         + kCallerMaximumBytes + 4 * 4096 * sizeof(arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrix));
    struct Charge final
    {
        std::shared_ptr<NativeCompositionRetentionBudget> budget;
        std::size_t bytes = 0;
        ~Charge() { std::lock_guard<std::mutex> lock(budget->mutex_); budget->bytes_ -= bytes; }
    };
    static std::shared_ptr<const Charge> reserve(
        const std::shared_ptr<NativeCompositionRetentionBudget>& budget,
        std::size_t bytes, std::string& error)
    {
        std::lock_guard<std::mutex> lock(budget->mutex_);
        if (bytes > kMaximumBytes - budget->bytes_)
        { error = "native composition caller and escaped retention exceed logical CPU capacity"; return {}; }
        // Construct with zero debit first so allocation failure cannot leave a
        // debit with no owner. The caller's existing exception policy applies.
        auto charge = std::make_shared<Charge>();
        charge->budget = budget; charge->bytes = bytes;
        budget->bytes_ += bytes;
        return charge;
    }
    std::size_t logicalBytes() const noexcept
    { std::lock_guard<std::mutex> lock(mutex_); return bytes_; }
    // Stack-only inspector storage. Reserve before scratch initialization and
    // release it after those values die, without allocating a new charge owner.
    class Scratch final
    {
    public:
        Scratch()=default;
        Scratch(const Scratch&)=delete;
        bool acquire(const std::shared_ptr<NativeCompositionRetentionBudget>& budget,std::size_t bytes) noexcept
        {
            std::lock_guard<std::mutex> lock(budget->mutex_);
            if (bytes>kMaximumBytes-budget->bytes_)return false;
            budget->bytes_+=bytes;budget_=budget.get();bytes_=bytes;return true;
        }
        ~Scratch()
        { if(budget_){std::lock_guard<std::mutex> lock(budget_->mutex_);budget_->bytes_-=bytes_;} }
    private:
        NativeCompositionRetentionBudget* budget_=nullptr;
        std::size_t bytes_=0;
    };
private:
    mutable std::mutex mutex_;
    std::size_t bytes_ = 0;
};
// A synchronous request borrows its fixed intent. Only the executor can seal a
// retained copy, together with the actual adapted sources and native owner.
class ImportedParentBindingOwner final
{
public:
    using Intent = visualimportedscenerender::ImportedParentIntentV1;
    using Scene = HarmonicMIDI::grid::Visual3DScene;
    struct Parents final { std::array<HarmonicMIDI::grid::SceneObjectId,64> values {}; };

    // Named logical units count fixed reads/tests, loop control and each record
    // probe separately. Byte comparisons/copies use the reachable capacities.
    // This does not measure library, allocator or native-driver internals.
    static bool sceneBytes(const Scene& scene, NativeSceneAnimationCompositionBudget& budget,
                           std::size_t& bytes) noexcept
    {
        // One vector capacity/addition and two string guards/additions in
        // every light slot. 32 setup units and 64 per-slot units cover those
        // fixed header reads, arithmetic, branches and counter updates.
        if (!budget.consumeWork(1,32 + 64*Scene::kMaxLights)) return false;
        bytes=sizeof(Scene);
        const auto add=[&](std::size_t count,std::size_t stride) {
            if (bytes>NativeSceneAnimationCompositionBudget::kMaximumBytes
                || count>(NativeSceneAnimationCompositionBudget::kMaximumBytes-bytes)/stride) return false;
            bytes+=count*stride;return true;
        };
        if (!add(scene.textureTexels.capacity(),sizeof(HarmonicMIDI::grid::SceneTexelRgba8))) return false;
        for (const auto& light:scene.lights)
        {
            const auto& a=light.environmentImage.asset.id;
            const auto& b=light.environmentImage.asset.contentSha256;
            if (a.capacity()>=NativeSceneAnimationCompositionBudget::kMaximumBytes
                || b.capacity()>=NativeSceneAnimationCompositionBudget::kMaximumBytes
                || !add(a.capacity()+1,1) || !add(b.capacity()+1,1)) return false;
        }
        return true;
    }

    // Reconstruct from the unchanged imported and authored procedural scenes.
    // Direction 2 proves the full root union; labels are the compiler's issued
    // value and are compared exactly again by the expected-request inspector.
    static bool parents(const Intent& intent,const Scene& imported,const Scene& procedural,
                        const Scene& basis,NativeSceneAnimationCompositionBudget& budget,
                        Parents& result) noexcept
    {
        if (!budget.consumeWork(1,24)) return false;
        const auto p=procedural.objectCount;
        const auto i=imported.objectCount;
        const auto n=basis.objectCount;
        const auto k=intent.rootMembershipCount;
        if (p>64 || i>64 || n>64 || p+i!=n || !p || !i || k>64) return false;
        // validImportedParentIntent visits all slots and its triangular duplicate
        // loop. Then complete-union lookup, expected-parent assembly and final
        // identity lookup visit their own tables. Reserve failures in full.
        const auto work=sizeof(Intent)*2 + 64*40 + 4*(k*(k-1)/2)
            +64+4 + 6*p + 8*k*p + 12*k +4*4
            +n + 10*n*i + 12*n*p + 12*n*k + n*32;
        if (!budget.consumeWork(work,1) || !visualimportedscenerender::validImportedParentIntent(intent)) return false;
        std::array<bool,64> seen {};
        std::array<std::size_t,4> groups {};
        if (intent.direction==2)
        {
            std::size_t roots=0;
            for (std::size_t x=0;x<p;++x) if (!procedural.objects[x].parent.isValid()) ++roots;
            if (roots!=k) return false;
            for (std::size_t x=0;x<k;++x)
            {
                const auto& entry=intent.rootMembership[x];
                std::size_t match=p;
                for (std::size_t y=0;y<p;++y)
                    if (procedural.objects[y].id.value==entry.objectId) {if (match!=p) return false;match=y;}
                if (match==p || seen[match] || procedural.objects[match].parent.isValid()) return false;
                seen[match]=true;++groups[entry.inputIndex];
            }
            for (std::size_t x=0;x<p;++x)
                if (!procedural.objects[x].parent.isValid() && !seen[x]) return false;
            for (std::size_t x=1;x<4;++x)
                if (bool(intent.connectedInputMask&(1u<<x))!=bool(groups[x])) return false;
        }
        else
        {
            bool found=false;
            for (std::size_t x=0;x<p;++x) if (procedural.objects[x].id.value==intent.proceduralAnchorId) found=true;
            if (!found || procedural.objects[0].id.value!=intent.proceduralAnchorId) return false;
        }
        const auto anchor=intent.direction==1
            ? HarmonicMIDI::grid::scenecomposition::proceduralObjectId({intent.proceduralAnchorId})->value
            : imported.objects[0].id.value;
        std::size_t added=0;
        for (std::size_t x=0;x<n;++x)
        {
            const auto& object=basis.objects[x];
            bool found=false;
            if (object.id.value<0x80000000u)
            {
                for (std::size_t y=0;y<i;++y) if (imported.objects[y].id==object.id)
                {
                    if (found || imported.objects[y].parent!=object.parent) return false;
                    found=true;
                }
            }
            else
            {
                for (std::size_t y=0;y<p;++y)
                    if ((procedural.objects[y].id.value|0x80000000u)==object.id.value)
                    {
                        const auto parent=procedural.objects[y].parent.value;
                        if (found || object.parent.value!=(parent?parent|0x80000000u:0)) return false;
                        found=true;
                    }
            }
            if (!found) return false;
            auto expected=object.parent;
            if (!expected.isValid())
            {
                if (intent.direction==1 && object.id.value<0x80000000u) {expected={anchor};++added;}
                if (intent.direction==2 && object.id.value>=0x80000000u)
                    for (std::size_t y=0;y<k;++y)
                        if ((intent.rootMembership[y].objectId|0x80000000u)==object.id.value
                            && (intent.requestedParentMask&(1u<<intent.rootMembership[y].inputIndex)))
                        {expected={anchor};++added;}
            }
            result.values[x]=expected;
        }
        return added>0;
    }

    static bool conservation(const Scene& basis,const Scene& scene,const Parents& parents,
                             NativeSceneAnimationCompositionBudget& budget) noexcept
    {
        std::size_t a=0,b=0;
        if (!sceneBytes(basis,budget,a) || !sceneBytes(scene,budget,b)
            || !budget.consumeWork(a+b,1) || !budget.consumeWork(64,sizeof(HarmonicMIDI::grid::SceneObjectRecord)+3)) return false;
        const auto same=[](const auto& x,const auto& y) {
            static_assert(std::is_trivially_copyable_v<std::decay_t<decltype(x)>>);
            return std::memcmp(&x,&y,sizeof(x))==0;
        };
        if (basis.schemaVersion!=scene.schemaVersion || basis.id!=scene.id || basis.activeCamera!=scene.activeCamera
            || !same(basis.ambientColor,scene.ambientColor) || basis.vertexCount!=scene.vertexCount
            || basis.indexCount!=scene.indexCount || basis.objectCount!=scene.objectCount
            || basis.materialCount!=scene.materialCount || basis.textureCount!=scene.textureCount
            || basis.textureTexelCount!=scene.textureTexelCount || basis.lightCount!=scene.lightCount
            || basis.cameraCount!=scene.cameraCount || !same(basis.vertices,scene.vertices)
            || !same(basis.indices,scene.indices)
            || !same(basis.textures,scene.textures) || !same(basis.cameras,scene.cameras)
            || basis.textureTexels.size()!=scene.textureTexels.size()) return false;
        if (basis.textureTexels.size() && std::memcmp(basis.textureTexels.data(),scene.textureTexels.data(),
                basis.textureTexels.size()*sizeof(HarmonicMIDI::grid::SceneTexelRgba8))) return false;
        for (std::size_t x=0;x<64;++x)
        {
            auto expected=basis.objects[x];
            if (x<basis.objectCount) expected.parent=parents.values[x];
            if (!same(expected,scene.objects[x])) return false;
        }
        // Compare fields, not material padding around its enum/bool. Inactive
        // records are conserved as well, including any float bit patterns.
        if (!budget.consumeWork(Scene::kMaxMaterials,24) || !budget.consumeWork(Scene::kMaxLights,128)) return false;
        for (std::size_t x=0;x<Scene::kMaxMaterials;++x)
        {
            const auto& a=basis.materials[x];const auto& b=scene.materials[x];
            if (a.id!=b.id || !same(a.baseColor,b.baseColor) || a.baseColorTexture!=b.baseColorTexture
                || a.metallicRoughnessTexture!=b.metallicRoughnessTexture || a.normalTexture!=b.normalTexture
                || a.occlusionTexture!=b.occlusionTexture || a.emissiveTexture!=b.emissiveTexture
                || !same(a.opacity,b.opacity) || !same(a.metallic,b.metallic) || !same(a.roughness,b.roughness)
                || !same(a.emissive,b.emissive) || !same(a.normalScale,b.normalScale)
                || !same(a.occlusionStrength,b.occlusionStrength) || a.alphaMode!=b.alphaMode
                || !same(a.alphaCutoff,b.alphaCutoff) || a.doubleSided!=b.doubleSided) return false;
        }
        for (std::size_t x=0;x<Scene::kMaxLights;++x)
            if (!HarmonicMIDI::grid::scenecomposition::sameLightRecord(basis.lights[x],scene.lights[x])) return false;
        return true;
    }

    static bool unchangedSource(const Scene& frozen,const Scene& actual,
                                NativeSceneAnimationCompositionBudget& budget) noexcept
    {
        // conservation visits all fixed records and reachable texels/identities.
        // Preserve original parent words here rather than applying new edges.
        if (!budget.consumeWork(1,24) || frozen.objectCount>64 || actual.objectCount>64
            || !budget.consumeWork(sizeof(Parents)+64*(2*sizeof(HarmonicMIDI::grid::SceneObjectId)+4),1)) return false;
        Parents original;
        for (std::size_t x=0;x<frozen.objectCount;++x) original.values[x]=frozen.objects[x].parent;
        return conservation(frozen,actual,original,budget);
    }

    bool inspect(const Intent& expected) const noexcept
    {
        NativeCompositionRetentionBudget::Scratch scratch;
        constexpr auto scratchBytes=2*sizeof(Parents)+64*sizeof(bool)+4*sizeof(std::size_t)
            +sizeof(HarmonicMIDI::grid::SceneObjectRecord)+sizeof(NativeCompositionRetentionBudget::Scratch)
            +8*sizeof(std::size_t)+2*sizeof(std::shared_ptr<const void>);
        if (!budget_ || !budget_->consumeWork(scratchBytes,1) || !retention_
            || !scratch.acquire(retention_->budget,scratchBytes)
            || !budget_->consumeWork(sizeof(Intent)*2,1)
            || !budget_->consumeWork(sizeof(*this),1)
            || !visualimportedscenerender::sameImportedParentIntent(expected,intent_)
            || !payload_ || !document_ || !animation_ || !imported_ || !procedural_ || !proceduralWitness_ || !basis_ || !scene_
            || !native_ || native_->scene()!=scene_ || native_->revision()!=structural_
            || native_->evaluationRevision()!=evaluation_ || native_->projectGeneration()!=project_
            || native_->helperGeneration()!=helper_) return false;
        Parents expectedParents;
        return unchangedSource(*proceduralWitness_,*procedural_,*budget_)
            && parents(expected,*imported_,*proceduralWitness_,*basis_,*budget_,expectedParents)
            && conservation(*basis_,*scene_,expectedParents,*budget_);
    }
private:
    friend class ImportedAnimatedScenePayloadExecution;
    friend class ImportedParentBindingFrame;
    ImportedParentBindingOwner()=default;
    ImportedParentBindingOwner(const ImportedParentBindingOwner&)=delete;
    // Release source/native ownership before returning the storage debit.
    std::shared_ptr<const NativeCompositionRetentionBudget::Charge> retention_;
    std::shared_ptr<NativeSceneAnimationCompositionBudget> budget_;
    Intent intent_;
    visualanimationimport::Request operation_;
    visualmodelassetpayload::PayloadPtr payload_;
    std::shared_ptr<const gltf::GlbStaticMeshDocument> document_;
    std::shared_ptr<const gltf::GlbAnimationDocument> animation_;
    std::shared_ptr<const Scene> imported_,procedural_,proceduralWitness_,basis_,scene_;
    std::shared_ptr<const arbitgpu::NativeDeformationCompositionOwner> native_;
    std::uint64_t structural_=0,evaluation_=0,project_=0,helper_=0;
    int renderClip_=0;
};

// Per-frame light records are part of the parent frame identity. A modulated
// light is a different frame, so reuse requires the exact same records.
inline bool sameParentLightRecords(const std::vector<HarmonicMIDI::grid::SceneLightRecord>& stored,
    const std::vector<HarmonicMIDI::grid::SceneLightRecord>& runtime) noexcept
{
    using HarmonicMIDI::grid::scenecomposition::sameLightRecord;
    return stored.size()==runtime.size()
        && std::equal(runtime.begin(),runtime.end(),stored.begin(),sameLightRecord);
}

class ImportedParentBindingFrame final
{
public:
    bool matches(const visualmodelassetpayload::PayloadPtr& payload,
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const ImportedAnimationDeformationEvaluation& evaluation,
        const videorender::animation3d::RenderedDeformationFrame& frame,
        const arbitgpu::NativeImportedSceneRuntimeInputs& runtime,
        ImportedAnimationEvaluationOwner use,std::optional<std::uint32_t> camera) const noexcept
    {
        if (!owner_ || !owner_->budget_->consumeWork(sizeof(*this),2)
            || !owner_->inspect(owner_->intent_)) return false;
        using HarmonicMIDI::grid::scenecomposition::sameCameraRecord;
        const auto sameCamera=[&](const auto& a,const auto& b) {
            return a.has_value()==b.has_value() && (!a || sameCameraRecord(*a,*b));
        };
        // Both light operands are admitted before nested string comparison.
        if (!owner_->budget_->consumeWork(lights_.size()+runtime.animatedLights.size(),
                4*sizeof(HarmonicMIDI::grid::SceneLightRecord))) return false;
        for (const auto& value:lights_)
            if (!owner_->budget_->consumeLightIdentityWork(value,2)) return false;
        for (const auto& value:runtime.animatedLights)
            if (!owner_->budget_->consumeLightIdentityWork(value,2)) return false;
        // Environment light and modulation overrides are stored and compared like the records.
        if (!owner_->budget_->consumeWork(runtime.lightOverrides.size()+overrides_.size(),
                sizeof(arbitgpu::NativeImportedLightOverride))) return false;
        if ((lightOverride_ && !owner_->budget_->consumeLightIdentityWork(*lightOverride_,2))
            || (runtime.lightOverride && !owner_->budget_->consumeLightIdentityWork(*runtime.lightOverride,2)))
            return false;
        const bool sameLightOverride=(!lightOverride_ && !runtime.lightOverride)
            || (lightOverride_ && runtime.lightOverride
                && HarmonicMIDI::grid::scenecomposition::sameLightRecord(*lightOverride_,*runtime.lightOverride));
        const bool sameLights=sameParentLightRecords(lights_,runtime.animatedLights);
        return payload==payload_ && source==source_ && source && source->scene==owner_->scene_
            && source->compositionOwner==owner_->native_ && source->structuralRevision==owner_->structural_
            && source->evaluationRevision==owner_->evaluation_ && evaluation.deformation==deformation_
            && evaluation.sceneAnimation==sample_ && evaluation.owner==use_ && use==use_
            && evaluation.playback.weight==playback_.weight
            && evaluation.playback.sample.timeSeconds==playback_.sample.timeSeconds
            && evaluation.playback.sample.playback==playback_.sample.playback
            && evaluation.playback.sample.rangeStartSeconds==playback_.sample.rangeStartSeconds
            && evaluation.playback.sample.rangeEndSeconds==playback_.sample.rangeEndSeconds
            && evaluation.sourceStableId==source->sourceStableId && evaluation.deformationStableId==source->deformationStableId
            && frame.source==source_ && frame.deformation==deformation_ && frame.nativeFrame==nativeFrame_
            && nativeFrame_ && nativeFrame_->width()==width_ && nativeFrame_->height()==height_
            && frame.use==(use_==ImportedAnimationEvaluationOwner::Preview
                ?videorender::animation3d::RenderUse::Preview:videorender::animation3d::RenderUse::Export)
            && frame.stats.time==time_ && frame.stats.sourceStableId==source->sourceStableId
            && frame.stats.deformationStableId==source->deformationStableId && frame.stats.revision==owner_->structural_
            && frame.stats.clipId==source->clip.value && runtime.compositionFrame==composition_
            && composition_ && composition_->owner()==owner_->native_ && composition_->retains(deformation_)
            && composition_->matchesTime(time_) && camera==cameraSelector_
            && sameCamera(runtime.cameraOverride,override_) && sameCamera(runtime.animatedCamera,camera_)
            && runtime.timeSeconds==timeSeconds_ && runtime.objectNodeStableId==node_
            && runtime.imageOutput==imageOutput_ && runtime.rawExportMask==rawExportMask_
            && runtime.linearColor==linearColor_ && samePrevious(runtime.previousMotion,previous_)
            && runtime.objectTranslationOffset==translation_ && runtime.objectRotationDegrees==rotation_
            && runtime.objectScale==scale_ && runtime.cameraTranslationOffset==cameraTranslation_
            && runtime.cameraRotationDegrees==cameraRotation_ && runtime.cameraVerticalFovDegrees==cameraFov_
            && runtime.morphWeight==morphWeight_ && runtime.emissionGain==emissionGain_
            && sameLightOverride && arbitgpu::sameNativeImportedLightOverrides(runtime.lightOverrides,overrides_)
            && sameLights && !runtime.noteInstanceMapping;
    }
    static bool samePrevious(const std::optional<arbitgpu::NativeSceneMotionSample>& a,
                             const std::optional<arbitgpu::NativeSceneMotionSample>& b) noexcept
    {
        using HarmonicMIDI::grid::scenecomposition::sameCameraRecord;
        return a.has_value()==b.has_value() && (!a || (a->owner==b->owner
            && a->deformation==b->deformation && a->compositionFrame==b->compositionFrame
            && a->importedNodeWorldMatrices==b->importedNodeWorldMatrices && a->noteTransforms==b->noteTransforms
            && a->objectTranslationOffset==b->objectTranslationOffset && a->objectRotationDegrees==b->objectRotationDegrees
            && a->objectScale==b->objectScale && a->cameraTranslationOffset==b->cameraTranslationOffset
            && a->cameraRotationDegrees==b->cameraRotationDegrees
            && a->cameraVerticalFovDegrees==b->cameraVerticalFovDegrees
            && a->morphWeight==b->morphWeight && a->cameraOverride.has_value()==b->cameraOverride.has_value()
            && a->timeSeconds==b->timeSeconds && a->vertexSpectrum==b->vertexSpectrum
            && a->scenePositions==b->scenePositions
            && (!a->cameraOverride || sameCameraRecord(*a->cameraOverride,*b->cameraOverride))));
    }
    static bool capacitiesFit(const visualanimationimport::Request& copy,const visualanimationimport::Request& original) noexcept
    {
        return copy.asset.id.capacity()<=original.asset.id.capacity()
            && copy.asset.contentSha256.capacity()<=original.asset.contentSha256.capacity()
            && copy.asset.sourceMediaType.capacity()<=original.asset.sourceMediaType.capacity()
            && copy.clipName.capacity()<=original.clipName.capacity()
            && copy.crossfadeClipName.capacity()<=original.crossfadeClipName.capacity();
    }
    static bool sameOperation(const visualanimationimport::Request& a,const visualanimationimport::Request& b) noexcept
    {
        const auto& x=a.playback;const auto& y=b.playback;
        const auto& p=a.pose;const auto& q=b.pose;
        return a.sourceStableId==b.sourceStableId && a.deformationStableId==b.deformationStableId
            && a.schedule==b.schedule && visualanimationimport::sameAsset(a.asset,b.asset)
            && a.sceneIndex==b.sceneIndex && a.meshStableId==b.meshStableId
            && a.animationClipStableId==b.animationClipStableId && a.crossfadeClipStableId==b.crossfadeClipStableId
            && a.clipName==b.clipName && a.crossfadeClipName==b.crossfadeClipName
            && a.crossfadeProgress==b.crossfadeProgress && a.combinationMode==b.combinationMode
            && x.timeSource==y.timeSource && x.timelineSeconds==y.timelineSeconds
            && x.clipStartTimelineSeconds==y.clipStartTimelineSeconds && x.timelineBeat==y.timelineBeat
            && x.clipStartBeat==y.clipStartBeat && x.beatsPerLoop==y.beatsPerLoop && x.speed==y.speed
            && x.offsetSeconds==y.offsetSeconds && x.trimStartSeconds==y.trimStartSeconds
            && x.trimEndSeconds==y.trimEndSeconds && x.weight==y.weight && x.playback==y.playback
            && p.meshStableId==q.meshStableId && p.boneStableId==q.boneStableId
            && p.morphTargetStableId==q.morphTargetStableId && p.morphTargetIndex==q.morphTargetIndex
            && p.boneEnabled==q.boneEnabled && p.morphEnabled==q.morphEnabled && p.translation==q.translation
            && p.rotationDegrees==q.rotationDegrees && p.scale==q.scale && p.morphWeight==q.morphWeight
            && p.nodeStableId==q.nodeStableId && p.morphWeightDeltas==q.morphWeightDeltas;
    }
    bool inspectExpected(const ImportedParentBindingOwner::Intent& expected,
        const visualdeformation::RationalFrameTime& time,std::uint64_t structural,std::uint64_t evaluation,
        std::uint64_t project,std::uint64_t helper,int clip,ImportedAnimationEvaluationOwner use,
        std::uint32_t width,std::uint32_t height,
        const visualanimationimport::Request* operation = nullptr,
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& procedural = {},
        std::optional<std::uint32_t> selector = {},
        const std::optional<HarmonicMIDI::grid::SceneCameraRecord>& override = {},
        const arbitgpu::NativeImportedSceneRuntimeInputs* runtime = nullptr) const noexcept
    {
        if (operation)
        {
            // All string operands are debited before exact selection/playback
            // equality, including rejected expected-request substitutions.
            for (const auto* value:{operation,&operation_})
                for (const auto* text:{&value->asset.id,&value->asset.contentSha256,&value->asset.sourceMediaType,
                                      &value->clipName,&value->crossfadeClipName})
                    if (!owner_ || text->capacity()>=NativeSceneAnimationCompositionBudget::kMaximumWork
                        || !owner_->budget_->consumeWork(text->capacity()+1,1)) return false;
            if (!owner_->budget_->consumeWork(1,48) || !sameOperation(*operation,operation_)
                || procedural!=owner_->procedural_ || selector!=cameraSelector_
                || override.has_value()!=override_.has_value()
                || (override && !HarmonicMIDI::grid::scenecomposition::sameCameraRecord(*override,*override_))) return false;
        }
        if (runtime && (!owner_ || !owner_->budget_->consumeWork(1,24)
            || runtime->imageOutput!=imageOutput_ || runtime->rawExportMask!=rawExportMask_
            || runtime->linearColor!=linearColor_ || runtime->objectTranslationOffset!=translation_
            || runtime->objectRotationDegrees!=rotation_ || runtime->objectScale!=scale_
            || runtime->cameraTranslationOffset!=cameraTranslation_ || runtime->emissionGain!=emissionGain_
            || runtime->cameraRotationDegrees!=cameraRotation_
            || runtime->cameraVerticalFovDegrees!=cameraFov_)) return false;
        return owner_ && owner_->inspect(expected) && owner_->budget_->consumeWork(1,10)
            && time==time_ && structural==owner_->structural_ && evaluation==owner_->evaluation_
            && project==owner_->project_ && helper==owner_->helper_ && clip==owner_->renderClip_
            && use==use_ && width==width_ && height==height_;
    }
private:
    friend class ImportedAnimatedScenePayloadExecution;
    ImportedParentBindingFrame()=default;
    ImportedParentBindingFrame(const ImportedParentBindingFrame&)=delete;
    std::shared_ptr<const void> retention_;
    std::shared_ptr<const ImportedParentBindingOwner> owner_;
    visualmodelassetpayload::PayloadPtr payload_;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source_;
    std::shared_ptr<const arbitgpu::NativeDeformationCompositionFrame> composition_;
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation_;
    std::shared_ptr<const visualanimation::Sample> sample_;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> nativeFrame_;
    std::optional<arbitgpu::NativeSceneMotionSample> previous_;
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> camera_,override_;
    std::optional<std::uint32_t> cameraSelector_;
    // Exact light records this frame was rendered with (see sameParentLightRecords).
    std::vector<HarmonicMIDI::grid::SceneLightRecord> lights_;
    // Environment light (request.light) and per-light modulation, stored like lights_.
    std::optional<HarmonicMIDI::grid::SceneLightRecord> lightOverride_;
    std::vector<arbitgpu::NativeImportedLightOverride> overrides_;
    visualanimationimport::Request operation_;
    visualanimation::ResolvedPlaybackControl playback_;
    visualdeformation::RationalFrameTime time_;
    ImportedAnimationEvaluationOwner use_=ImportedAnimationEvaluationOwner::Preview;
    renderpassoutput::Output imageOutput_=renderpassoutput::Output::Color;
    std::uint32_t rawExportMask_=0,width_=0,height_=0;
    std::uint64_t node_=0;
    std::array<float,3> translation_ {},rotation_ {},cameraTranslation_ {},cameraRotation_ {};
    float scale_=1,morphWeight_=1,emissionGain_=1,cameraFov_=0;
    float timeSeconds_=0;
    bool linearColor_=false;
};
} // namespace videohelper::modelpayload
