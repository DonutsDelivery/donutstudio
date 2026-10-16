#pragma once

#include "imported_animation_deformation_consumer.h"
#include "gpu_backend/backend.h"
#include "../../shared/VisualModelAssetPayload.h"
#include "../../shared/ImportedLightControl.h"
#include "../../shared/Visual3DSceneComposition.h"

#include <algorithm>
#include <mutex>
#include "glb_native_render_seam.h"
#include "../../shared/VisualImportedObjectLocalTransform.h"
#include "../../shared/VisualImportedSceneIdentity.h"

namespace videohelper::modelpayload
{
class ImportedAnimatedScenePayloadExecution;
class ImportedScenePayloadExecution;

// Separate logical CPU allowance for camera/light owner associations. Native
// images, allocator metadata and driver memory remain owned by the renderer.
class NativeSceneAnimationCompositionBudget final
{
public:
    static constexpr std::size_t kMaximumBytes = 8 * 1024 * 1024;
    static constexpr std::uint64_t kMaximumWork = 4ull * 1024 * 1024 * 1024;
    struct Charge final
    {
        std::shared_ptr<NativeSceneAnimationCompositionBudget> budget;
        std::size_t bytes = 0;
        ~Charge()
        {
            if (!budget) return;
            std::lock_guard<std::mutex> lock(budget->mutex_);
            budget->bytes_ -= bytes;
        }
    };
    bool work(std::uint64_t count, std::uint64_t steps, std::string& error)
    {
        if (consumeWork(count,steps)) return true;
        error = "camera/light composition named work exceeds capacity";
        return false;
    }
    bool consumeWork(std::uint64_t count, std::uint64_t steps) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!steps || count > (kMaximumWork - work_) / steps)
        {
            work_ = kMaximumWork;
            return false;
        }
        work_ += count * steps;
        return true;
    }
    // Strings exist even in inactive light slots and version-zero images.
    // Use capacity plus the terminator for prospective copies and retention.
    static bool addLightIdentityBytes(const HarmonicMIDI::grid::SceneLightRecord& light,
        std::size_t copies, std::size_t& bytes, std::string& error)
    {
        const auto add=[&](const std::string& text) {
            if (!copies || bytes>kMaximumBytes || text.capacity()>=kMaximumBytes
                || text.capacity()+1>(kMaximumBytes-bytes)/copies)
            { error="camera/light composition light identity exceeds logical CPU capacity";return false; }
            bytes+=(text.capacity()+1)*copies;return true;
        };
        return add(light.environmentImage.asset.id) && add(light.environmentImage.asset.contentSha256);
    }
    bool consumeLightIdentityWork(const HarmonicMIDI::grid::SceneLightRecord& light,
        std::uint64_t steps) noexcept
    {
        const auto consume=[&](const std::string& text) {
            if (text.capacity()>=kMaximumWork) return consumeWork(kMaximumWork+1,1);
            return consumeWork(text.capacity()+1,steps);
        };
        return consume(light.environmentImage.asset.id) && consume(light.environmentImage.asset.contentSha256);
    }
    static std::shared_ptr<const Charge> reserve(
        const std::shared_ptr<NativeSceneAnimationCompositionBudget>& budget,
        std::size_t bytes, std::string& error)
    {
        std::lock_guard<std::mutex> lock(budget->mutex_);
        if (bytes > kMaximumBytes - budget->bytes_)
        { error = "camera/light composition escaped retention exceeds logical CPU capacity"; return {}; }
        auto charge = std::make_shared<Charge>();
        charge->budget = budget; charge->bytes = bytes;
        budget->bytes_ += bytes;
        return charge;
    }
    std::size_t logicalBytes() const noexcept
    { std::lock_guard<std::mutex> lock(mutex_); return bytes_; }
    std::uint64_t logicalWork() const noexcept
    { std::lock_guard<std::mutex> lock(mutex_); return work_; }
private:
    mutable std::mutex mutex_;
    std::size_t bytes_ = sizeof(NativeSceneAnimationCompositionBudget);
    std::uint64_t work_ = 0;
};

// Source-node and admitted-scene checks exclude the adapter's synthetic camera.
inline const HarmonicMIDI::grid::SceneCameraRecord* sourceCameraRecord(
    const gltf::GlbStaticMeshDocument& document,
    const HarmonicMIDI::grid::Visual3DScene& scene, std::uint32_t node) noexcept
{
    if (!node || node > document.nodes.size() || document.nodes.size() > 16384
        || scene.cameraCount > scene.cameras.size()) return nullptr;
    const auto definition = document.nodes[node - 1].camera;
    if (!definition || *definition >= document.cameras.size()) return nullptr;
    // Only records adapted from selected-scene roots appear in this scene.
    const auto end = scene.cameras.begin() + scene.cameraCount;
    const auto camera = std::find_if(scene.cameras.begin(), end,
        [node](const auto& record) { return record.id.value == node; });
    return camera == end ? nullptr : &*camera;
}

// Issued only by the payload executor after exact selection and ancestry checks.
// It never requires or fabricates a deformation asset, joint or snapshot.
class NativeSceneAnimationCompositionOwner final
{
public:
    enum class Mode { CameraLightOnlyLegacy, StaticLocalHierarchy, RigidLocalScene };
    Mode mode() const noexcept { return mode_; }
    struct RecordMapping final
    {
        enum class Kind { Camera, Light } kind = Kind::Camera;
        std::uint32_t original = 0, composed = 0;
        bool imported = false;
    };
    const auto& payload() const noexcept { return payload_; }
    const auto& document() const noexcept { return document_; }
    const auto& animation() const noexcept { return animation_; }
    const auto& importedScene() const noexcept { return imported_; }
    const auto& proceduralScene() const noexcept { return procedural_; }
    const auto& scene() const noexcept { return scene_; }
    const auto& selection() const noexcept { return selection_; }
    const auto& sourceCameraNodeId() const noexcept { return sourceCameraNodeId_; }
    const auto& source() const noexcept { return source_; }
    const auto& recordMappings() const noexcept { return mappings_; }
    std::size_t recordMappingCount() const noexcept { return mappingCount_; }
    std::uint64_t structuralRevision() const noexcept { return structural_; }
    std::uint64_t evaluationRevision() const noexcept { return evaluation_; }
    std::uint64_t projectGeneration() const noexcept { return project_; }
    std::uint64_t helperGeneration() const noexcept { return helper_; }
    int renderClipId() const noexcept { return renderClip_; }
private:
    friend class ImportedAnimatedScenePayloadExecution;
    friend class ImportedScenePayloadExecution;
    friend class NativeSceneAnimationCompositionFrame;
    NativeSceneAnimationCompositionOwner() = default;
    explicit NativeSceneAnimationCompositionOwner(ImportedScenePayloadExecution&) noexcept
        : mode_(Mode::StaticLocalHierarchy) {}
    NativeSceneAnimationCompositionOwner(const NativeSceneAnimationCompositionOwner&) = delete;
    NativeSceneAnimationCompositionOwner& operator=(const NativeSceneAnimationCompositionOwner&) = delete;
    // Declared first so every retained payload dies before its debit is refunded.
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> charge_;
    Mode mode_ = Mode::CameraLightOnlyLegacy;
    std::shared_ptr<const gltf::GlbNativeSceneAdmission> staticAdmission_;
    std::uint64_t sourceStableId_ = 0, renderStableId_ = 0;
    visualmodelassetpayload::PayloadPtr payload_;
    std::shared_ptr<const gltf::GlbStaticMeshDocument> document_;
    std::shared_ptr<const gltf::GlbAnimationDocument> animation_;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> imported_, procedural_, scene_;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source_;
    visualanimationimport::Request selection_;
    std::optional<std::uint32_t> sourceCameraNodeId_;
    std::array<RecordMapping, HarmonicMIDI::grid::Visual3DScene::kMaxCameras
        + HarmonicMIDI::grid::Visual3DScene::kMaxLights> mappings_ {};
    std::size_t mappingCount_ = 0;
    std::uint64_t structural_ = 0, evaluation_ = 0, project_ = 0, helper_ = 0;
    int renderClip_ = 0;
};

// The actual executor seals selected-camera provenance for true deformation.
// Receipt copies retain this private association even if public fields vanish.
class NativeSelectedDeformationCameraFrame final
{
public:
    bool matches(const visualmodelassetpayload::PayloadPtr& payload,
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const ImportedAnimationDeformationEvaluation& evaluation,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& native,
        ImportedAnimationEvaluationOwner use, visualdeformation::RationalFrameTime time,
        const arbitgpu::NativeImportedSceneRuntimeInputs& runtime,
        std::optional<std::uint32_t> selector,
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& renderedSource,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& renderedDeformation,
        std::uint8_t renderUse) const noexcept
    {
        // Failed validation consumes attempted work too. Camera records contain
        // fixed scalar fields only, with no nested allocation or string scan.
        if (!charge_ || !charge_->budget->consumeWork(sizeof(*this),4)) return false;
        using HarmonicMIDI::grid::scenecomposition::sameCameraRecord;
        return selector && *selector == selector_ && payload == payload_ && source == source_
            && renderedSource == source_ && renderedDeformation == deformation_
            && renderUse == (use_ == ImportedAnimationEvaluationOwner::Preview ? 0u : 1u)
            && source && evaluation.deformation == deformation_ && deformation_
            && evaluation.sceneAnimation == sample_ && evaluation.owner == use_ && use == use_
            && evaluation.sourceStableId == source->sourceStableId
            && evaluation.deformationStableId == source->deformationStableId
            && deformation_->revision() == source->structuralRevision
            && deformation_->clip() == source->clip && deformation_->time() == time_
            && time == time_ && runtime.timeSeconds == timeSeconds_
            && runtime.animatedCamera && sameCameraRecord(*runtime.animatedCamera,camera_)
            && runtime.cameraOverride.has_value() == override_.has_value()
            && (!override_ || sameCameraRecord(*runtime.cameraOverride,*override_))
            && native && native == nativeFrame_ && native->width() == width_ && native->height() == height_;
    }
private:
    friend class ImportedAnimatedScenePayloadExecution;
    NativeSelectedDeformationCameraFrame() = default;
    NativeSelectedDeformationCameraFrame(const NativeSelectedDeformationCameraFrame&) = delete;
    NativeSelectedDeformationCameraFrame& operator=(const NativeSelectedDeformationCameraFrame&) = delete;
    // Refund only after every retained original owner has been released.
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> charge_;
    visualmodelassetpayload::PayloadPtr payload_;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source_;
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation_;
    std::shared_ptr<const visualanimation::Sample> sample_;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> nativeFrame_;
    HarmonicMIDI::grid::SceneCameraRecord camera_;
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> override_;
    visualdeformation::RationalFrameTime time_;
    ImportedAnimationEvaluationOwner use_ = ImportedAnimationEvaluationOwner::Preview;
    std::uint32_t selector_ = 0, width_ = 0, height_ = 0;
    float timeSeconds_ = 0;
};

class NativeSceneAnimationCompositionFrame final
{
public:
    const auto& owner() const noexcept { return owner_; }
    const auto& sample() const noexcept { return sample_; }
    const auto& nativeFrame() const noexcept { return nativeFrame_; }
    const auto& time() const noexcept { return time_; }
    const auto& operation() const noexcept { return operation_; }
    const auto& lights() const noexcept { return lights_; }
    const auto& camera() const noexcept { return camera_; }
    const auto& sampledCameras() const noexcept { return cameras_; }
    ImportedAnimationEvaluationOwner use() const noexcept { return use_; }
    using Matrix = std::array<float, 16>;
    const auto& localTransform() const noexcept { return localTransform_; }
    // Checked worlds of the drawn objects, without this frame's overlay. Null
    // unless a single-leaf candidate built them during staging.
    const auto& objectWorlds() const noexcept { return objectWorlds_; }
    std::uint64_t overlayNodeStableId() const noexcept { return overlayNodeStableId_; }
    // Each query uses the sealed frame's own overlay, including a predecessor.
    // No scene copy, world table, animation sample or identity fallback exists.
    bool checkedStaticWorld(const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        HarmonicMIDI::grid::SceneObjectId object, Matrix& output, std::string& error,
        bool finalPlacement = true) const
    {
        using Mode = NativeSceneAnimationCompositionOwner::Mode;
        if (!owner_ || owner_->mode() != Mode::StaticLocalHierarchy || scene != owner_->scene()
            || !charge_ || !owner_->document())
        { error = "static local world has a foreign issuing scene"; return false; }
        const auto count = scene->objectCount;
        const auto nodes = owner_->document()->nodes.size();
        if (count > scene->objects.size() || nodes > 88
            || !charge_->budget->work(count * count, kVisit, error)
            || !charge_->budget->work(count + 6, kProduct, error)
            || !charge_->budget->work(2, kInverse, error)
            || !charge_->budget->work(count + nodes, kVisit, error)
            || !charge_->budget->work(sizeof(*this), 4, error)) return false;
        Matrix world;
        if (!originalWorld(*scene, object, world))
        { error = "static local draw identity or hierarchy is invalid"; return false; }
        const auto* original = arbitgpu::findFixtureObject(*owner_->importedScene(), object);
        if (original)
        {
            const auto node = (object.value - 1u)
                / HarmonicMIDI::grid::visualimportedsceneidentity::kObjectPrimitiveStride;
            bool affected = false;
            if (!descendant(node, affected))
            { error = "static local draw source ancestry is invalid"; return false; }
            if (affected) world = multiply(correction_, world);
        }
        Matrix inverse;
        if (!inverseAffine(world, inverse))
        { error = "static local authored world is singular or nonfinite"; return false; }
        if (finalPlacement && (!overlayNodeStableId_ || (original
            && (object.value - 1u) / HarmonicMIDI::grid::visualimportedsceneidentity::kObjectPrimitiveStride
                + 1u == overlayNodeStableId_))) world = multiply(overlay_, world);
        if (!inverseAffine(world, inverse))
        { error = "static local final world is singular or nonfinite"; return false; }
        output = world; return true;
    }
    // Candidate binding only. Native device/context and completed-frame sealing
    // remain required at the guarded P4 submission boundary.
    bool matchesStaticCandidate(const visualmodelassetpayload::PayloadPtr& payload,
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        std::uint64_t source, std::uint64_t render, std::uint64_t structural,
        std::uint64_t evaluation, std::uint64_t project, std::uint64_t helper,
        int clip, ImportedAnimationEvaluationOwner use, visualdeformation::RationalFrameTime time,
        std::uint32_t width, std::uint32_t height,
        const visualimportedobjectlocal::ObjectLocalTransformV1& local,
        const arbitgpu::NativeImportedSceneRuntimeInputs& runtime,
        std::optional<std::uint32_t> sourceCameraNodeId = {}) const noexcept
    {
        using namespace HarmonicMIDI::grid::scenecomposition;
        if (!charge_ || !charge_->budget->consumeWork(sizeof(*this), 4)
            || !charge_->budget->consumeWork(runtime.animatedLights.size(), 4*sizeof(HarmonicMIDI::grid::SceneLightRecord))
            || !charge_->budget->consumeWork(runtime.lightOverrides.size()+lightOverrides_.size(),
                sizeof(arbitgpu::NativeImportedLightOverride))) return false;
        for (const auto& value : runtime.animatedLights)
            if (!charge_->budget->consumeLightIdentityWork(value, 2)) return false;
        for (const auto& value : lights_)
            if (!charge_->budget->consumeLightIdentityWork(value, 2)) return false;
        if ((runtime.lightOverride && !charge_->budget->consumeLightIdentityWork(*runtime.lightOverride, 2))
            || (lightOverride_ && !charge_->budget->consumeLightIdentityWork(*lightOverride_, 2))) return false;
        const auto camera = [](const auto& a, const auto& b) {
            return a.has_value() == b.has_value() && (!a || sameCameraRecord(*a, *b));
        };
        const auto light = [](const auto& a, const auto& b) {
            return a.has_value() == b.has_value() && (!a || sameLightRecord(*a, *b));
        };
        return owner_ && owner_->mode() == NativeSceneAnimationCompositionOwner::Mode::StaticLocalHierarchy
            && !sample_ && !owner_->animation() && owner_->document() && owner_->staticAdmission_
            && owner_->sourceCameraNodeId() == sourceCameraNodeId && sourceCameraNodeId_ == sourceCameraNodeId
            && owner_->payload() == payload && owner_->scene() == scene
            && owner_->sourceStableId_ == source && owner_->renderStableId_ == render
            && owner_->structuralRevision() == structural && owner_->evaluationRevision() == evaluation
            && owner_->projectGeneration() == project && owner_->helperGeneration() == helper
            && owner_->renderClipId() == clip && use_ == use && time_ == time
            && width_ == width && height_ == height
            && visualimportedobjectlocal::same(localTransform_, local)
            && runtime.objectNodeStableId == overlayNodeStableId_
            && runtime.objectTranslationOffset == overlayTranslation_
            && runtime.objectRotationDegrees == overlayRotation_ && runtime.objectScale == overlayScale_
            && runtime.cameraTranslationOffset == cameraTranslation_
            && runtime.cameraRotationDegrees == cameraRotation_ && runtime.cameraVerticalFovDegrees == cameraFov_
            && runtime.timeSeconds == timeSeconds_ && runtime.imageOutput == imageOutput_
            && runtime.rawExportMask == rawExportMask_ && runtime.linearColor == linearColor_
            && camera(runtime.animatedCamera, camera_) && camera(runtime.cameraOverride, cameraOverride_)
            && light(runtime.lightOverride, lightOverride_) && runtime.animatedLights.size() == lights_.size()
            && std::equal(runtime.animatedLights.begin(), runtime.animatedLights.end(), lights_.begin(), sameLightRecord)
            && arbitgpu::sameNativeImportedLightOverrides(runtime.lightOverrides, lightOverrides_)
            && arbitgpu::sameNativeImportedObjectWorlds(objectWorlds_, runtime.objectWorlds)
            && !runtime.compositionFrame && runtime.localSceneFrame.get() == this;
    }
    bool chargeCallerWork(std::uint64_t count, std::uint64_t steps, std::string& error) const
    {
        if (!charge_ || !callerBudget_)
        { error="static local caller has no issuing budget"; return false; }
        return charge_->budget->work(count,steps,error);
    }
    std::shared_ptr<const void> reserveCaller(std::size_t bytes, std::string& error) const
    {
        if (!charge_ || !callerBudget_ || !reserveCaller_)
        { error="static local caller has no issuing budget"; return {}; }
        if (!charge_->budget->work(1,kVisit,error)) return {};
        if (bytes>callerMaximumBytes_)
        { error="static local caller retention exceeds its existing cap"; return {}; }
        if (!charge_->budget->work(bytes,2,error)) return {};
        return reserveCaller_(callerBudget_,bytes,error);
    }
    bool matches(const visualmodelassetpayload::PayloadPtr& payload,
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const ImportedAnimationDeformationEvaluation& evaluation,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& native,
        ImportedAnimationEvaluationOwner use, visualdeformation::RationalFrameTime time,
        const arbitgpu::NativeImportedSceneRuntimeInputs& runtime,
        std::optional<std::uint32_t> sourceCameraNodeId = {}) const noexcept
    {
        using namespace HarmonicMIDI::grid::scenecomposition;
        // Revalidation is real caller work even when a source/use/time check
        // fails. It shares the authority's permanent attempted-work allowance.
        if (!charge_) return false;
        const auto& budget=charge_->budget;
        if (!budget->consumeWork(sizeof(*this),4)
            || !budget->consumeWork(lights_.size(),4*sizeof(HarmonicMIDI::grid::SceneLightRecord))
            || !budget->consumeWork(runtime.animatedLights.size(),4*sizeof(HarmonicMIDI::grid::SceneLightRecord))
            || !budget->consumeWork(runtime.lightOverrides.size()+lightOverrides_.size(),
                sizeof(arbitgpu::NativeImportedLightOverride))) return false;
        // Both operands are admitted before sameLightRecord can compare nested
        // strings. A substituted runtime or failed match never refunds work.
        if ((lightOverride_ && !budget->consumeLightIdentityWork(*lightOverride_,2))
            || (runtime.lightOverride && !budget->consumeLightIdentityWork(*runtime.lightOverride,2))) return false;
        for (const auto& value:lights_)
            if (!budget->consumeLightIdentityWork(value,2)) return false;
        for (const auto& value:runtime.animatedLights)
            if (!budget->consumeLightIdentityWork(value,2)) return false;
        const auto camera=[](const auto& a,const auto& b) {
            return a.has_value()==b.has_value() && (!a || sameCameraRecord(*a,*b));
        };
        const auto light=[](const auto& a,const auto& b) {
            return a.has_value()==b.has_value() && (!a || sameLightRecord(*a,*b));
        };
        return owner_ && owner_->mode() == NativeSceneAnimationCompositionOwner::Mode::CameraLightOnlyLegacy
            && sample_ && owner_->sourceCameraNodeId() == sourceCameraNodeId
            && sourceCameraNodeId_ == sourceCameraNodeId
            && (!sourceCameraNodeId || (camera_ && camera_->id.value == *sourceCameraNodeId))
            && owner_->payload() == payload && owner_->source() == source
            && source && source->scene == owner_->scene() && !source->deformation
            && !source->compositionOwner && source->draws.empty()
            && source->sourceStableId == operation_.sourceStableId
            && source->deformationStableId == operation_.deformationStableId
            && source->structuralRevision == owner_->structuralRevision()
            && source->evaluationRevision == owner_->evaluationRevision()
            && !evaluation.deformation && evaluation.sceneAnimation == sample_
            && evaluation.owner == use_ && use == use_ && time == time_
            && source->clip == sample_->clip()
            && evaluation.sourceStableId == operation_.sourceStableId
            && evaluation.deformationStableId == operation_.deformationStableId
            && runtime.timeSeconds == timeSeconds_ && runtime.imageOutput == imageOutput_
            && runtime.rawExportMask == rawExportMask_ && runtime.linearColor == linearColor_
            && !runtime.previousMotion && !runtime.compositionFrame && !runtime.localSceneFrame
            && camera(runtime.animatedCamera,camera_) && camera(runtime.cameraOverride,cameraOverride_)
            && light(runtime.lightOverride,lightOverride_) && runtime.animatedLights.size() == lights_.size()
            && std::equal(runtime.animatedLights.begin(),runtime.animatedLights.end(),lights_.begin(),sameLightRecord)
            && arbitgpu::sameNativeImportedLightOverrides(runtime.lightOverrides,lightOverrides_)
            && native && native == nativeFrame_ && native->width() == width_ && native->height() == height_;
    }
private:
    friend class ImportedAnimatedScenePayloadExecution;
    friend class ImportedScenePayloadExecution;
    friend class NativeSceneAnimationCompositionOwner;
    explicit NativeSceneAnimationCompositionFrame(ImportedScenePayloadExecution&) noexcept {}
    NativeSceneAnimationCompositionFrame() = default;
    NativeSceneAnimationCompositionFrame(const NativeSceneAnimationCompositionFrame&) = delete;
    NativeSceneAnimationCompositionFrame& operator=(const NativeSceneAnimationCompositionFrame&) = delete;
    static constexpr std::uint64_t kProduct = 4096, kInverse = 4096, kVisit = 64;
    static Matrix multiply(const Matrix& a, const Matrix& b) noexcept
    {
        Matrix result {};
        for (std::size_t column = 0; column < 4; ++column)
            for (std::size_t row = 0; row < 4; ++row)
                for (std::size_t inner = 0; inner < 4; ++inner)
                    result[column * 4 + row] += a[inner * 4 + row] * b[column * 4 + inner];
        return result;
    }
    static Matrix compose(const HarmonicMIDI::grid::SceneTransform3D& transform) noexcept
    {
        const auto x = transform.rotation.x, y = transform.rotation.y;
        const auto z = transform.rotation.z, w = transform.rotation.w;
        const auto sx = transform.scale.x, sy = transform.scale.y, sz = transform.scale.z;
        return {(1-2*(y*y+z*z))*sx, 2*(x*y+w*z)*sx, 2*(x*z-w*y)*sx, 0,
            2*(x*y-w*z)*sy, (1-2*(x*x+z*z))*sy, 2*(y*z+w*x)*sy, 0,
            2*(x*z+w*y)*sz, 2*(y*z-w*x)*sz, (1-2*(x*x+y*y))*sz, 0,
            transform.translation.x, transform.translation.y, transform.translation.z, 1};
    }
    static bool nearAxisValue(float a, float b) noexcept
    { return std::abs(a-b) <= .0001f * std::max({1.0f, std::abs(a), std::abs(b)}); }
    static bool sameMatrix(const Matrix& a, const Matrix& b) noexcept
    { for (std::size_t i=0; i<a.size(); ++i) if (!nearAxisValue(a[i], b[i])) return false; return true; }
    static bool inverseAffine(const Matrix& value, Matrix& result) noexcept
    {
        if (!std::all_of(value.begin(), value.end(), [](float x) { return std::isfinite(x); }) || value[3] != 0 || value[7] != 0 || value[11] != 0 || value[15] != 1)
            return false;
        const double a=value[0], b=value[4], c=value[8], d=value[1], e=value[5], f=value[9];
        const double g=value[2], h=value[6], i=value[10];
        const double determinant=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
        if (!std::isfinite(determinant) || determinant == 0) return false;
        result = {float((e*i-f*h)/determinant), float((f*g-d*i)/determinant), float((d*h-e*g)/determinant), 0,
            float((c*h-b*i)/determinant), float((a*i-c*g)/determinant), float((b*g-a*h)/determinant), 0,
            float((b*f-c*e)/determinant), float((c*d-a*f)/determinant), float((a*e-b*d)/determinant), 0,
            0,0,0,1};
        for (std::size_t row=0; row<3; ++row)
            result[12+row] = -(result[row]*value[12]+result[4+row]*value[13]+result[8+row]*value[14]);
        return std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); });
    }

    static bool decompose(const std::array<float, 16>& matrix,
                   HarmonicMIDI::grid::SceneTransform3D& transform)
    {
        Matrix inverse;
        if (!inverseAffine(matrix, inverse))
            return false;

        transform.translation = {matrix[12], matrix[13], matrix[14]};
        std::array<float, 3> x {matrix[0], matrix[1], matrix[2]};
        std::array<float, 3> y {matrix[4], matrix[5], matrix[6]};
        std::array<float, 3> z {matrix[8], matrix[9], matrix[10]};
        const auto length = [] (const std::array<float, 3>& value)
        {
            return std::sqrt(value[0] * value[0] + value[1] * value[1]
                             + value[2] * value[2]);
        };
        auto sx = length(x), sy = length(y), sz = length(z);
        if (!std::isfinite(sx) || !std::isfinite(sy) || !std::isfinite(sz)
            || sx <= 0.000001f || sy <= 0.000001f || sz <= 0.000001f)
            return false;
        for (auto& value : x) value /= sx;
        for (auto& value : y) value /= sy;
        for (auto& value : z) value /= sz;

        const auto dot = [] (const auto& left, const auto& right)
        {
            return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
        };
        if (!nearAxisValue(dot(x, y), 0.0f) || !nearAxisValue(dot(x, z), 0.0f)
            || !nearAxisValue(dot(y, z), 0.0f))
            return false;

        auto determinant = x[0] * (y[1] * z[2] - y[2] * z[1])
                         - y[0] * (x[1] * z[2] - x[2] * z[1])
                         + z[0] * (x[1] * y[2] - x[2] * y[1]);
        if (!nearAxisValue(std::abs(determinant), 1.0f))
            return false;
        if (determinant < 0.0f)
        {
            sx = -sx;
            for (auto& value : x) value = -value;
            determinant = -determinant;
        }
        if (!nearAxisValue(determinant, 1.0f))
            return false;

        const auto r00 = x[0], r01 = y[0], r02 = z[0];
        const auto r10 = x[1], r11 = y[1], r12 = z[1];
        const auto r20 = x[2], r21 = y[2], r22 = z[2];
        HarmonicMIDI::grid::SceneQuaternion rotation;
        const auto trace = r00 + r11 + r22;
        if (trace > 0.0f)
        {
            const auto scale = 2.0f * std::sqrt(trace + 1.0f);
            rotation.w = 0.25f * scale;
            rotation.x = (r21 - r12) / scale;
            rotation.y = (r02 - r20) / scale;
            rotation.z = (r10 - r01) / scale;
        }
        else if (r00 > r11 && r00 > r22)
        {
            const auto scale = 2.0f * std::sqrt(1.0f + r00 - r11 - r22);
            rotation.w = (r21 - r12) / scale;
            rotation.x = 0.25f * scale;
            rotation.y = (r01 + r10) / scale;
            rotation.z = (r02 + r20) / scale;
        }
        else if (r11 > r22)
        {
            const auto scale = 2.0f * std::sqrt(1.0f + r11 - r00 - r22);
            rotation.w = (r02 - r20) / scale;
            rotation.x = (r01 + r10) / scale;
            rotation.y = 0.25f * scale;
            rotation.z = (r12 + r21) / scale;
        }
        else
        {
            const auto scale = 2.0f * std::sqrt(1.0f + r22 - r00 - r11);
            rotation.w = (r10 - r01) / scale;
            rotation.x = (r02 + r20) / scale;
            rotation.y = (r12 + r21) / scale;
            rotation.z = 0.25f * scale;
        }

        const auto quaternionLength = std::sqrt(rotation.x * rotation.x
                                              + rotation.y * rotation.y
                                              + rotation.z * rotation.z
                                              + rotation.w * rotation.w);
        if (!std::isfinite(quaternionLength) || quaternionLength <= 0.000001f)
            return false;
        rotation.x /= quaternionLength;
        rotation.y /= quaternionLength;
        rotation.z /= quaternionLength;
        rotation.w /= quaternionLength;
        if (rotation.w < 0.0f)
        {
            rotation.x = -rotation.x;
            rotation.y = -rotation.y;
            rotation.z = -rotation.z;
            rotation.w = -rotation.w;
        }
        transform.rotation = rotation;
        transform.scale = {sx, sy, sz};
        return sameMatrix(matrix, compose(transform));
    }

    static bool originalWorld(const HarmonicMIDI::grid::Visual3DScene& scene,
        HarmonicMIDI::grid::SceneObjectId object, Matrix& output) noexcept
    {
        const auto* record = arbitgpu::findFixtureObject(scene, object);
        if (!record) return false;
        auto result = compose(record->transform);
        auto parent = record->parent;
        for (std::size_t depth=0; parent.isValid(); ++depth)
        {
            if (depth >= scene.objectCount) return false;
            record = arbitgpu::findFixtureObject(scene, parent);
            if (!record) return false;
            result = multiply(compose(record->transform), result);
            parent = record->parent;
        }
        output = result; return true;
    }
    bool descendant(std::size_t node, bool& affected) const noexcept
    {
        const auto& nodes = owner_->document()->nodes;
        affected = false;
        for (std::size_t depth=0; depth<nodes.size(); ++depth)
        {
            if (node >= nodes.size()) return false;
            if (node + 1u == localTransform_.nodeStableId) { affected = true; return true; }
            if (!nodes[node].parent) return true;
            node = *nodes[node].parent;
        }
        return false;
    }
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> charge_;
    Matrix correction_ {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    Matrix overlay_ {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    visualimportedobjectlocal::ObjectLocalTransformV1 localTransform_;
    std::shared_ptr<const arbitgpu::NativeImportedObjectWorlds> objectWorlds_;
    std::uint64_t overlayNodeStableId_ = 0;
    std::array<float,3> overlayTranslation_ {}, overlayRotation_ {};
    float overlayScale_ = 1;
    std::shared_ptr<const void> callerBudget_;
    std::shared_ptr<const void> (*reserveCaller_)(const std::shared_ptr<const void>&, std::size_t, std::string&) = nullptr;
    std::size_t callerMaximumBytes_ = 0;
    std::array<float,3> cameraTranslation_ {};
    std::array<float,3> cameraRotation_ {};
    float cameraFov_ = 0;
    std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame> staticNativeFrame_;
    std::shared_ptr<const NativeSceneAnimationCompositionOwner> owner_;
    std::shared_ptr<const visualanimation::Sample> sample_;
    // Actual GPU frame retains the backend resources and device/context lease.
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> nativeFrame_;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> material_;
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> camera_, cameraOverride_;
    std::optional<HarmonicMIDI::grid::SceneLightRecord> lightOverride_;
    std::vector<HarmonicMIDI::grid::SceneLightRecord> lights_;
    // Applied to lights_ after sampling. Stored so a changed override never reuses this frame.
    std::vector<arbitgpu::NativeImportedLightOverride> lightOverrides_;
    std::vector<HarmonicMIDI::grid::SceneCameraRecord> cameras_;
    importedlightcontrol::Controls controls_;
    visualanimationimport::Request operation_;
    std::optional<std::uint32_t> sourceCameraNodeId_;
    visualanimation::ResolvedPlaybackControl playback_;
    visualdeformation::RationalFrameTime time_;
    ImportedAnimationEvaluationOwner use_ = ImportedAnimationEvaluationOwner::Preview;
    std::uint32_t width_ = 0, height_ = 0;
    float timeSeconds_ = 0;
    renderpassoutput::Output imageOutput_ = renderpassoutput::Output::Color;
    std::uint32_t rawExportMask_ = 0;
    bool linearColor_ = false;
};
} // namespace videohelper::modelpayload
