#include "glb_native_render_seam.h"
#include "material_program_compiler.h"
#include "native_scene_animation_composition.h"
#include <type_traits>
#include <limits>
#include <new>
#include "surface_material_binding_admission.h"
#include "gpu_backend/fixture_smooth_glass.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

namespace videohelper::gltf
{
namespace
{
using HarmonicMIDI::grid::Visual3DScene;

GlbAdmissionOptions nativeRenderOptions (
    std::optional<std::size_t> sceneIndex)
{
    GlbAdmissionOptions options;
    auto& limits = options.limits;
    limits.maxContainerBytes = GlbNativeRenderSeam::kMaxSourceBytes;
    limits.maxJsonBytes = GlbNativeRenderSeam::kMaxJsonBytes;
    limits.maxBinBytes = GlbNativeRenderSeam::kMaxBinBytes;
    limits.maxScenes = 64;
    limits.maxNodes = Visual3DScene::kMaxObjects
                    + Visual3DScene::kMaxCameras
                    + Visual3DScene::kMaxLights;
    limits.maxNodeDepth = 64;
    limits.maxMeshes = Visual3DScene::kMaxObjects;
    limits.maxPrimitives = Visual3DScene::kMaxObjects;
    limits.maxMaterials = Visual3DScene::kMaxMaterials;
    limits.maxTextures = Visual3DScene::kMaxTextures;
    limits.maxImages = Visual3DScene::kMaxTextures;
    limits.maxSamplers = Visual3DScene::kMaxTextures;
    limits.maxCameras = Visual3DScene::kMaxCameras;
    limits.maxLights = Visual3DScene::kMaxLights;
    limits.maxDecodedVertices = Visual3DScene::kMaxVertices;
    limits.maxDecodedIndices = Visual3DScene::kMaxIndices;
    limits.maxDecodedBytes = Visual3DScene::kMaxVertices
                           * sizeof (HarmonicMIDI::grid::SceneVertex)
                           + Visual3DScene::kMaxIndices * sizeof (std::uint32_t)
                           + Visual3DScene::kMaxTextureTexels
                               * sizeof (HarmonicMIDI::grid::SceneTexelRgba8);
    limits.maxEmbeddedImageBytes = Visual3DScene::kMaxTextureTexels * 4u;
    limits.maxDecodedImageBytes = Visual3DScene::kMaxTextureTexels * 4u;
    limits.maxImageWidth = Visual3DScene::kMaxTextureDimension;
    limits.maxImageHeight = Visual3DScene::kMaxTextureDimension;
    options.admitAnimations = true;
    options.admitSkins = true;
    options.supportedRequiredExtensions = {"KHR_lights_punctual"};
    options.sceneIndex = sceneIndex;
    return options;
}

// Counts reachable standard-library capacities, excluding allocator metadata.
bool retainedDocumentBytes(const GlbStaticMeshDocument& document,
    const std::shared_ptr<modelpayload::NativeSceneAnimationCompositionBudget>& budget,
    std::size_t& bytes, std::string& error)
{
    using Budget = modelpayload::NativeSceneAnimationCompositionBudget;
    bytes = sizeof(document) + sizeof(Budget::Charge);
    const auto add = [&](std::size_t count, std::size_t stride) {
        if (!stride || bytes > Budget::kMaximumBytes
            || count > (Budget::kMaximumBytes - bytes) / stride)
        { error = "static retained document exceeds logical CPU capacity"; return false; }
        const auto lanes = count * stride;
        if (!budget->work(lanes, 2, error)) return false;
        bytes += lanes; return true;
    };
    const auto vector = [&](const auto& values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        return add(values.capacity(), sizeof(Value));
    };
    const auto text = [&](const std::string& value) {
        return value.capacity() != std::numeric_limits<std::size_t>::max()
            && add(value.capacity() + 1, 1);
    };
    if (!budget->work(bytes, 2, error) || !text(document.metadata.generator)
        || !vector(document.metadata.extensionsUsed) || !vector(document.metadata.extensionsRequired)
        || !vector(document.bufferViews) || !vector(document.accessors) || !vector(document.samplers)
        || !vector(document.images) || !vector(document.textures) || !vector(document.materials)
        || !vector(document.meshes) || !vector(document.cameras) || !vector(document.lights)
        || !vector(document.nodes) || !vector(document.scenes)) return false;
    for (const auto& value : document.metadata.extensionsUsed) if (!text(value)) return false;
    for (const auto& value : document.metadata.extensionsRequired) if (!text(value)) return false;
    for (const auto& value : document.samplers) if (!text(value.name)) return false;
    for (const auto& value : document.images)
        if (!text(value.name) || !text(value.mimeType) || !vector(value.decodedRgba8)) return false;
    for (const auto& value : document.textures) if (!text(value.name)) return false;
    for (const auto& value : document.materials) if (!text(value.name)) return false;
    for (const auto& value : document.cameras) if (!text(value.name)) return false;
    for (const auto& value : document.lights) if (!text(value.name)) return false;
    for (const auto& value : document.nodes) if (!text(value.name) || !vector(value.children)) return false;
    for (const auto& value : document.scenes) if (!text(value.name) || !vector(value.rootNodes)) return false;
    for (const auto& mesh : document.meshes)
    {
        if (!text(mesh.name) || !vector(mesh.primitives)) return false;
        for (const auto& value : mesh.primitives)
            if (!vector(value.positions) || !vector(value.normals) || !vector(value.tangents)
                || !vector(value.texCoords0) || !vector(value.colors0) || !vector(value.indices)
                || !vector(value.cornerSources)) return false;
    }
    return true;
}

// Both fixed scene values overlap during adapter return/shared retention. The
// old bounded decoder document remains temporary until the second reservation.
struct ColdRetention final
{
    std::shared_ptr<const modelpayload::NativeSceneAnimationCompositionBudget::Charge> fixed, closure;
    std::shared_ptr<const GlbStaticMeshDocument> document;
    std::shared_ptr<const Visual3DScene> scene;
};

bool coldClosureBytes(const GlbStaticMeshDocument& document,
    const std::shared_ptr<modelpayload::NativeSceneAnimationCompositionBudget>& budget,
    std::size_t& bytes, std::size_t& documentBytes, std::size_t& texels, std::size_t& textureGrowthOverlapTexels, std::string& error)
{
    using Budget = modelpayload::NativeSceneAnimationCompositionBudget;
    if (!retainedDocumentBytes(document, budget, bytes, error)) return false;
    documentBytes = bytes;
    const auto add = [&](std::size_t count, std::size_t stride) {
        if (!stride || bytes > Budget::kMaximumBytes || count > (Budget::kMaximumBytes-bytes)/stride)
        { error = "static cold admission exceeds logical CPU capacity"; return false; }
        if (!budget->work(count, 2*stride, error)) return false;
        bytes += count*stride; return true;
    };
    const auto text = [&](const std::string& value) { return value.capacity() < Budget::kMaximumBytes
        && add(value.capacity()+1, 1); };
    if (!text(document.metadata.generator)
        || !add(document.metadata.extensionsUsed.capacity()+document.metadata.extensionsRequired.capacity(), sizeof(std::string))) return false;
    for (const auto* list : {&document.metadata.extensionsUsed, &document.metadata.extensionsRequired})
        for (const auto& value : *list) if (!text(value)) return false;
    // Match copyMaterials' order and five roles. A repeated texture shares its
    // first adapted range; unused textures produce no scene texel allocation.
    // reserve grows to each prefix. Its largest retired capacity overlaps the
    // final new range while the selected vector implementation relocates it.
    texels = 0; textureGrowthOverlapTexels = 0;
    std::array<bool, Visual3DScene::kMaxTextures> copied {};
    if (!budget->work(document.textures.size()*document.materials.size(), 5*64, error)) return false;
    for (const auto& material : document.materials)
        for (const auto* info : {&material.baseColorTexture, &material.metallicRoughnessTexture,
                                &material.normalTexture, &material.occlusionTexture, &material.emissiveTexture})
        {
            if (!*info) continue;
            const auto index = (*info)->texture;
            if (index >= document.textures.size() || index >= copied.size())
            { error = "static cold texture index is invalid"; return false; }
            if (copied[index]) continue;
            const auto& texture = document.textures[index];
            if (texture.source >= document.images.size())
            { error = "static cold texture source is invalid"; return false; }
            const auto count = document.images[texture.source].decodedRgba8.size()/4;
            if (count > Visual3DScene::kMaxTextureTexels-texels)
            { error = "static cold texture closure exceeds the existing scene bound"; return false; }
            textureGrowthOverlapTexels = texels;
            texels += count;
            copied[index] = true;
        }
    // Existing adapter scratch: selected/state/retained-node bytes, its DFS
    // Frame's three size_t fields and one adapted ID per source texture.
    if (!add(document.nodes.size(), 3*sizeof(std::uint8_t))
        || !add(std::min(document.nodes.size(), GlbLimits{}.maxNodeDepth), 3*sizeof(std::size_t))
        || !add(document.textures.size(), sizeof(HarmonicMIDI::grid::SceneTextureId))) return false;
    if (!add(texels, sizeof(HarmonicMIDI::grid::SceneTexelRgba8))
        || !add(textureGrowthOverlapTexels, sizeof(HarmonicMIDI::grid::SceneTexelRgba8))) return false;
    std::size_t lights = 0;
    if (!budget->work(document.nodes.size(), 64, error)) return false;
    for (const auto& node : document.nodes) if (node.light) ++lights;
    const auto nodeBytes = GlbNativeRenderSeam::lightDefinitionNodeBytes();
    if (lights && !nodeBytes) { error = "static light map structural storage has no selected-library census"; return false; }
    return !lights || add(lights, nodeBytes);
}

bool fail (std::string& error, const char* stage, std::string diagnostic)
{
    error = stage;
    error += ": ";
    error += diagnostic.empty() ? "request rejected without a diagnostic"
                                : std::move (diagnostic);
    return false;
}
} // namespace

std::size_t GlbNativeRenderSeam::lightDefinitionNodeBytes() noexcept
{
#if defined(__GLIBCXX__)
    using Value = importedscene::LightDefinitions::value_type;
    // bits/stl_tree.h: _Rb_tree_node stores the base links/color and aligned value.
    return sizeof(std::_Rb_tree_node<Value>);
#else
    // Other library layouts require their source-owned node-storage boundary.
    // Only budgeted active candidates with source lights refuse here.
    return 0;
#endif
}

struct GlbNativeFrameReceipt::LocalRetention final
{
    std::shared_ptr<const void> charge;
    std::shared_ptr<const modelpayload::NativeSceneAnimationCompositionFrame> frame;
};
namespace
{
constexpr std::uint64_t kLocalReceiptLightIdentityVisitWork = 64;

bool localReceiptBytes(const GlbNativeFrameReceipt& source, std::size_t& bytes, std::string& error)
{
    using Budget = modelpayload::NativeSceneAnimationCompositionBudget;
    bytes = sizeof(source);
    const auto add = [&](std::size_t count, std::size_t stride) {
        if (!stride || bytes > Budget::kMaximumBytes || count > (Budget::kMaximumBytes-bytes)/stride)
        { error = "static local receipt copies exceed logical CPU capacity"; return false; }
        bytes += count*stride; return true;
    };
    const auto vector = [&](const auto& value) {
        return add(value.capacity(),sizeof(typename std::decay_t<decltype(value)>::value_type));
    };
    const auto text = [&](const std::string& value) { return value.capacity() < Budget::kMaximumBytes && add(value.capacity()+1,1); };
    if (!vector(source.draws) || !vector(source.materials) || !vector(source.textures)
        || !vector(source.textureRoles) || !vector(source.cameras) || !vector(source.lights) || !text(source.nativeBackend)) return false;
    for (const auto& light : source.lights)
        if (!text(light.environmentImage.asset.id) || !text(light.environmentImage.asset.contentSha256)) return false;
    return true;
}
bool localReceiptCapacitiesFit(const GlbNativeFrameReceipt& copy, const GlbNativeFrameReceipt& source) noexcept
{
    if (copy.draws.capacity()>source.draws.capacity() || copy.materials.capacity()>source.materials.capacity()
        || copy.textures.capacity()>source.textures.capacity() || copy.textureRoles.capacity()>source.textureRoles.capacity()
        || copy.cameras.capacity()>source.cameras.capacity() || copy.lights.capacity()>source.lights.capacity()
        || copy.nativeBackend.capacity()>source.nativeBackend.capacity()) return false;
    for (std::size_t i=0; i<copy.lights.size(); ++i)
        if (copy.lights[i].environmentImage.asset.id.capacity()>source.lights[i].environmentImage.asset.id.capacity()
            || copy.lights[i].environmentImage.asset.contentSha256.capacity()>source.lights[i].environmentImage.asset.contentSha256.capacity()) return false;
    return true;
}
}
void GlbNativeFrameReceipt::exchange(GlbNativeFrameReceipt& other) noexcept
{
    if (this == &other) return;
    // Three header writes per swap, two held-retention headers, and one
    // outgoing header disposal. This fixed allowance covers each active side;
    // it does not admit a native transaction or census its retained closure.
    constexpr auto fixedExchangeWork = 3*sizeof(GlbNativeFrameReceipt);
    constexpr auto fixedRetentionTransferWork = 2*sizeof(LocalRetention);
    constexpr auto fixedOutgoingDisposalWork = sizeof(GlbNativeFrameReceipt);
    const auto record = [](const std::shared_ptr<const LocalRetention>& retention) noexcept {
        if (!retention) return;
        std::string error;
        try
        {
            retention->frame->chargeCallerWork(1,
                fixedExchangeWork+fixedRetentionTransferWork+fixedOutgoingDisposalWork, error);
        }
        catch (const std::bad_alloc&)
        {
            // work() has already consumed or saturated the issuing ledger
            // before its refusal diagnostic can allocate. Owned values still
            // transfer unconditionally, including at that saturation boundary.
        }
    };
    record(localRetention_);
    record(other.localRetention_);
    using std::swap;
    swap(supportedSubset,other.supportedSubset);
    swap(downstreamHandoffComplete,other.downstreamHandoffComplete);
    swap(attributes,other.attributes);
    swap(draws,other.draws);
    swap(materials,other.materials);
    swap(textures,other.textures);
    swap(textureRoles,other.textureRoles);
    swap(cameras,other.cameras);
    swap(lights,other.lights);
    swap(activeCamera,other.activeCamera);
    swap(nativeBackend,other.nativeBackend);
    swap(nativeResourceCacheIdentity,other.nativeResourceCacheIdentity);
    swap(nativeDeviceOrContextIdentity,other.nativeDeviceOrContextIdentity);
    swap(nativeRendererGeneration,other.nativeRendererGeneration);
    swap(rendered.use,other.rendered.use);
    swap(rendered.sceneId,other.rendered.sceneId);
    swap(rendered.dimensions,other.rendered.dimensions);
    swap(rendered.stats,other.rendered.stats);
    rendered.nativeFrame.swap(other.rendered.nativeFrame);
    localRetention_.swap(other.localRetention_);
}
void GlbNativeFrameReceipt::copyValues(const GlbNativeFrameReceipt& other)
{
    supportedSubset=other.supportedSubset;
    downstreamHandoffComplete=other.downstreamHandoffComplete;
    attributes=other.attributes;
    draws=other.draws;
    materials=other.materials;
    textures=other.textures;
    textureRoles=other.textureRoles;
    cameras=other.cameras;
    lights=other.lights;
    activeCamera=other.activeCamera;
    nativeBackend=other.nativeBackend;
    nativeResourceCacheIdentity=other.nativeResourceCacheIdentity;
    nativeDeviceOrContextIdentity=other.nativeDeviceOrContextIdentity;
    nativeRendererGeneration=other.nativeRendererGeneration;
    rendered=other.rendered;
}
GlbNativeFrameReceipt::GlbNativeFrameReceipt(const GlbNativeFrameReceipt& other)
{
    if (other.localRetention_)
    {
        std::size_t bytes=0; std::string error;
        if (!other.localRetention_->frame->chargeCallerWork(1,2*sizeof(other),error)
            || !other.localRetention_->frame->chargeCallerWork(other.lights.size(),kLocalReceiptLightIdentityVisitWork,error)
            || !localReceiptBytes(other,bytes,error)) throw std::bad_alloc();
        auto charge=other.localRetention_->frame->reserveCaller(bytes+sizeof(LocalRetention),error);
        if (!charge) throw std::bad_alloc();
        localRetention_=std::make_shared<LocalRetention>(LocalRetention{std::move(charge),other.localRetention_->frame});
    }
    copyValues(other);
    if (localRetention_ && !localReceiptCapacitiesFit(*this,other)) throw std::bad_alloc();
}
GlbNativeFrameReceipt& GlbNativeFrameReceipt::operator=(const GlbNativeFrameReceipt& other)
{
    if (this!=&other)
    {
        // An inactive destination owns ordinary public value storage. Charge
        // its actual overlap during an active-local assignment before copying.
        std::shared_ptr<const void> overlap;
        if (other.localRetention_ && !localRetention_)
        {
            std::size_t bytes=0; std::string error;
            if (!other.localRetention_->frame->chargeCallerWork(1,2*sizeof(*this),error)
                || !other.localRetention_->frame->chargeCallerWork(lights.size(),kLocalReceiptLightIdentityVisitWork,error)
                || !localReceiptBytes(*this,bytes,error)) throw std::bad_alloc();
            overlap=other.localRetention_->frame->reserveCaller(bytes,error);
            if (!overlap) throw std::bad_alloc();
        }
        GlbNativeFrameReceipt candidate(other);
        exchange(candidate);
    }
    return *this;
}
GlbNativeFrameReceipt::GlbNativeFrameReceipt(GlbNativeFrameReceipt&& other) noexcept { exchange(other); }
GlbNativeFrameReceipt& GlbNativeFrameReceipt::operator=(GlbNativeFrameReceipt&& other) noexcept
{
    if (this!=&other) { GlbNativeFrameReceipt candidate(std::move(other)); exchange(candidate); }
    return *this;
}
std::shared_ptr<const modelpayload::NativeSceneAnimationCompositionFrame> GlbNativeFrameReceipt::localFrame() const noexcept
{ return localRetention_ ? localRetention_->frame : nullptr; }
bool GlbNativeFrameReceipt::prepareLocal(
    const std::shared_ptr<const modelpayload::NativeSceneAnimationCompositionFrame>& frame,
    const Visual3DScene& scene, const std::string& backend, std::string& error)
{
    using Budget = modelpayload::NativeSceneAnimationCompositionBudget;
    if (!frame || localRetention_ || scene.objectCount>scene.objects.size() || scene.materialCount>scene.materials.size()
        || scene.textureCount>scene.textures.size() || scene.cameraCount>scene.cameras.size() || scene.lightCount>scene.lights.size()) return false;
    // Census header and per-light identity probes are separate named visits,
    // debited before either proportional loop, including refused attempts.
    if (!frame->chargeCallerWork(1,2*sizeof(*this),error)
        || !frame->chargeCallerWork(scene.lightCount,kLocalReceiptLightIdentityVisitWork,error)
        || !frame->chargeCallerWork(lights.size(),kLocalReceiptLightIdentityVisitWork,error)) return false;
    std::size_t bytes=sizeof(*this)+sizeof(LocalRetention);
    const auto add=[&](std::size_t count,std::size_t stride) {
        if (!stride || bytes>Budget::kMaximumBytes || count>(Budget::kMaximumBytes-bytes)/stride)
        { error="static local prospective receipt exceeds logical CPU capacity";return false; }
        bytes+=count*stride;return true;
    };
    const auto text=[&](const std::string& value) {return value.capacity()<Budget::kMaximumBytes && add(value.capacity()+1,1);};
    if (!add(scene.objectCount,sizeof(GlbNativeDrawReceipt)) || !add(scene.materialCount,sizeof(GlbNativeMaterialReceipt))
        || !add(scene.textureCount,sizeof(GlbNativeTextureReceipt)) || !add(5*scene.materialCount,sizeof(GlbNativeTextureRoleReceipt))
        || !add(scene.cameraCount,sizeof(HarmonicMIDI::grid::SceneCameraRecord))
        || !add(scene.lightCount,sizeof(HarmonicMIDI::grid::SceneLightRecord)) || !text(backend)) return false;
    for (std::size_t i=0;i<scene.lightCount;++i)
        if (!text(scene.lights[i].environmentImage.asset.id) || !text(scene.lights[i].environmentImage.asset.contentSha256)) return false;
    // Old public values and the candidate overlap until publication. Reserve
    // the actual old capacities too; failure leaves the destination unchanged.
    std::size_t oldBytes=0;
    if (!localReceiptBytes(*this,oldBytes,error) || !add(oldBytes,1)) return false;
    try
    {
        auto charge=frame->reserveCaller(bytes,error); if (!charge) return false;
        GlbNativeFrameReceipt candidate;
        candidate.localRetention_=std::make_shared<LocalRetention>(LocalRetention{std::move(charge),frame});
        candidate.draws.reserve(scene.objectCount); candidate.materials.reserve(scene.materialCount);
        candidate.textures.reserve(scene.textureCount); candidate.textureRoles.reserve(5*scene.materialCount);
        candidate.cameras.reserve(scene.cameraCount); candidate.lights.reserve(scene.lightCount);
        candidate.nativeBackend.reserve(backend.capacity());
        if (candidate.draws.capacity()>scene.objectCount || candidate.materials.capacity()>scene.materialCount
            || candidate.textures.capacity()>scene.textureCount || candidate.textureRoles.capacity()>5*scene.materialCount
            || candidate.cameras.capacity()>scene.cameraCount || candidate.lights.capacity()>scene.lightCount
            || candidate.nativeBackend.capacity()>backend.capacity())
        { error="static local receipt capacity exceeds its prospective reservation";return false; }
        exchange(candidate);
    }
    catch (const std::bad_alloc&)
    { error="static local receipt allocation failed"; return false; }
    return true;
}

GlbNativeRenderSeam::GlbNativeRenderSeam (
    arbitgpu::NativeFixtureSceneBackend& backend) noexcept
    : backend_ (backend), renderer_ (backend)
{
}

bool GlbNativeRenderSeam::admit (
    const std::uint8_t* bytes,
    std::size_t size,
    std::optional<std::size_t> sceneIndex,
    GlbNativeSceneAdmission& output,
    std::string& error,
    std::uint32_t objectNodeStableId,
    std::uint32_t objectMeshStableId,
    std::shared_ptr<modelpayload::NativeSceneAnimationCompositionBudget> documentBudget, bool optionalDocumentRetention) const
{
    if (bytes == nullptr || size == 0)
        return fail (error, "GLB decode rejected", "source bytes are empty");
    if (size > kMaxSourceBytes)
        return fail (error, "GLB decode rejected", "source exceeds the 16 MiB native-render bound");

    using Budget = modelpayload::NativeSceneAnimationCompositionBudget;
    std::shared_ptr<const Budget::Charge> fixedCharge, closureCharge;
    std::size_t closureBytes = 0, documentBytes = 0, prospectiveTexels = 0, prospectiveTextureGrowthOverlapTexels = 0;
    if (documentBudget && !optionalDocumentRetention)
    {
        // Named cold work precedes decoder/parser/image visits. Source bytes
        // cover container/JSON/BIN scans; admitted decoded lanes and record
        // ceilings cover expansion, original hierarchy and adapter products.
        const auto limits = nativeRenderOptions(sceneIndex).limits;
        constexpr std::uint64_t coldSourceVisit = 64, coldDecodedByteVisit = 8,
            coldHierarchyVisit = 64, coldDecodedRecordVisit = 4096;
        if (!documentBudget->work(size, coldSourceVisit, error)
            || !documentBudget->work(limits.maxDecodedBytes, coldDecodedByteVisit, error)
            || !documentBudget->work(limits.maxNodes*limits.maxNodes, coldHierarchyVisit, error)
            || !documentBudget->work(limits.maxNodes, 6*4096+2*4096+coldHierarchyVisit, error)
            || !documentBudget->work(limits.maxDecodedVertices+limits.maxDecodedIndices, coldDecodedRecordVisit, error)) return false;
        const std::string empty;
        const auto fixedBytes = 2*sizeof(Visual3DScene)+sizeof(GlbNativeSceneAdmission)
            +sizeof(ColdRetention)+sizeof(Budget::Charge)
            +4*Visual3DScene::kMaxLights*(empty.capacity()+1);
        fixedCharge = Budget::reserve(documentBudget, fixedBytes, error);
        if (!fixedCharge) return false;
    }
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ++GlbColdAdmissionProbe::state().decodes;
#endif
    std::string diagnostic;
    auto decoded = decodeAnimatedGlbBaseScene (
        bytes, size, nativeRenderOptions (sceneIndex), diagnostic);
    if (! decoded)
        return fail (error, "GLB decode rejected", std::move (diagnostic));

    if (documentBudget && !optionalDocumentRetention)
    {
        if (!coldClosureBytes(*decoded, documentBudget, closureBytes, documentBytes, prospectiveTexels, prospectiveTextureGrowthOverlapTexels, error)) return false;
        closureCharge = Budget::reserve(documentBudget, closureBytes, error);
        if (!closureCharge) return false;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
        GlbColdAdmissionProbe::state().textureTexels = prospectiveTexels;
        GlbColdAdmissionProbe::state().textureGrowthOverlapTexels = prospectiveTextureGrowthOverlapTexels;
        GlbColdAdmissionProbe::state().reservedBytes = documentBudget->logicalBytes();
#endif
    }
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ++GlbColdAdmissionProbe::state().adaptations;
#endif
    auto scene = adaptStaticGlbToVisual3DScene (*decoded, diagnostic);
    if (! scene)
        return fail (error, "GLB scene adaptation rejected", std::move (diagnostic));

    if ((objectNodeStableId == 0) != (objectMeshStableId == 0)
        || (objectNodeStableId != 0 && (!sceneIndex
            || selectedImportedPrototype(*decoded, *scene, objectNodeStableId, objectMeshStableId) == nullptr)))
        return fail(error, "GLB object selection rejected", "node and mesh do not identify a drawn object in the exact source scene");

    if (documentBudget && !optionalDocumentRetention && scene->textureTexels.capacity() > prospectiveTexels)
        return fail(error, "GLB scene adaptation rejected", "texture capacity exceeds prospective cold admission");
    GlbNativeSceneAdmission admitted;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ++GlbColdAdmissionProbe::state().sharedScenes;
#endif
    admitted.scene = std::make_shared<const Visual3DScene> (std::move (*scene));
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ++GlbColdAdmissionProbe::state().metadataCopies;
#endif
    admitted.metadata = decoded->metadata;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ++GlbColdAdmissionProbe::state().lightMaps;
#endif
    admitted.importedLightDefinitions = importedscene::importedLightDefinitions(*decoded, *admitted.scene);
    admitted.selectedScene = decoded->selectedScene;
    admitted.sourceBytes = size;
    admitted.decodedVertexCount = decoded->decodedVertexCount;
    admitted.decodedIndexCount = decoded->decodedIndexCount;
    admitted.decodedBytes = decoded->decodedBytes;
    if (documentBudget && !optionalDocumentRetention)
    {
        if (admitted.metadata.generator.capacity() > decoded->metadata.generator.capacity()
            || admitted.metadata.extensionsUsed.capacity() > decoded->metadata.extensionsUsed.capacity()
            || admitted.metadata.extensionsRequired.capacity() > decoded->metadata.extensionsRequired.capacity())
            return fail(error, "GLB admission rejected", "metadata capacity exceeds prospective cold admission");
        const std::pair<const std::vector<std::string>*, const std::vector<std::string>*> identities[] = {
            {&admitted.metadata.extensionsUsed, &decoded->metadata.extensionsUsed},
            {&admitted.metadata.extensionsRequired, &decoded->metadata.extensionsRequired}};
        for (const auto& pair : identities)
            for (std::size_t i=0; i<pair.first->size(); ++i)
                if ((*pair.first)[i].capacity() > (*pair.second)[i].capacity())
                    return fail(error, "GLB admission rejected", "metadata string exceeds prospective cold admission");
        auto retained = std::make_shared<ColdRetention>();
        retained->fixed = fixedCharge; retained->closure = closureCharge;
        retained->scene = std::move(admitted.scene);
        retained->document = std::make_shared<const GlbStaticMeshDocument>(std::move(*decoded));
        admitted.retainedDocumentBytes = documentBytes;
        admitted.documentRetention = retained;
        admitted.scene = {retained, retained->scene.get()};
        admitted.document = {retained, retained->document.get()};
        admitted.sourceScene=admitted.scene;
    }
    if (documentBudget && optionalDocumentRetention)
    {
        // Legacy admission remains successful when the additional retained
        // document cannot fit. A later local edit then refuses without decoding.
        try
        {
            std::size_t retainedBytes=0; std::string retentionError;
            if (retainedDocumentBytes(*decoded,documentBudget,retainedBytes,retentionError)
                && retainedBytes<=Budget::kMaximumBytes-sizeof(Visual3DScene)
                && admitted.scene->textureTexels.capacity()<=(Budget::kMaximumBytes-retainedBytes-sizeof(Visual3DScene))/sizeof(HarmonicMIDI::grid::SceneTexelRgba8))
            {
                documentBytes=retainedBytes;
                retainedBytes+=sizeof(Visual3DScene)+admitted.scene->textureTexels.capacity()*sizeof(HarmonicMIDI::grid::SceneTexelRgba8);
                bool identities=documentBudget->work(sizeof(Visual3DScene),2,retentionError);
                for (const auto& light:admitted.scene->lights)
                    identities=identities && Budget::addLightIdentityBytes(light,1,retainedBytes,retentionError);
                auto charge=identities ? Budget::reserve(documentBudget,retainedBytes+sizeof(ColdRetention),retentionError) : nullptr;
                if (charge)
                {
                    auto retained=std::make_shared<ColdRetention>();
                    retained->closure=std::move(charge);
                    retained->document=std::make_shared<const GlbStaticMeshDocument>(std::move(*decoded));
                    retained->scene=std::move(admitted.scene);
                    admitted.retainedDocumentBytes=documentBytes;
                    admitted.documentRetention=retained;
                    admitted.document={retained,retained->document.get()};
                    admitted.scene={retained,retained->scene.get()};
                    admitted.sourceScene=admitted.scene;
                }
            }
        }
        catch (const std::bad_alloc&)
        {
            // The optional local prerequisite must not change v0 admission.
        }
    }
    output = std::move (admitted);
    error.clear();
    return true;
}

const HarmonicMIDI::grid::SceneObjectRecord* GlbNativeRenderSeam::selectedImportedPrototype(
    const GlbStaticMeshDocument& document, const Visual3DScene& scene,
    std::uint32_t nodeStableId, std::uint32_t meshStableId) noexcept
{
    if (nodeStableId == 0 || meshStableId == 0
        || nodeStableId > document.nodes.size() || meshStableId > document.meshes.size()
        || scene.objectCount > scene.objects.size()
        || document.nodes[nodeStableId - 1u].mesh != std::optional<std::size_t>{meshStableId - 1u})
        return nullptr;
    const auto end = scene.objects.begin() + scene.objectCount;
    const auto object = std::find_if(scene.objects.begin(), end,
        [nodeStableId](const auto& candidate) { return candidate.indexCount != 0
            && (candidate.id.value - 1u) / HarmonicMIDI::grid::visualimportedsceneidentity::kObjectPrimitiveStride
                + 1u == nodeStableId; });
    return object == end ? nullptr : &*object;
}

const HarmonicMIDI::grid::SceneCameraRecord* GlbNativeRenderSeam::selectedSourceCamera(
    const GlbNativeSceneAdmission& admission, std::uint32_t node) noexcept
{
    const auto& source=admission.sourceScene ? admission.sourceScene : admission.scene;
    return admission.document && source
        ? modelpayload::sourceCameraRecord(*admission.document, *source, node) : nullptr;
}

bool GlbNativeRenderSeam::renderPreview (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    return renderPreview(admission, dimensions, {}, output, error);
}

bool GlbNativeRenderSeam::renderPreview (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    return render (admission, dimensions,
                   videorender::fixture3d::RenderUse::Preview,
                   sceneInputs, output, error);
}

bool GlbNativeRenderSeam::renderExport (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    return renderExport(admission, dimensions, {}, output, error);
}

bool GlbNativeRenderSeam::renderExport (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    return render (admission, dimensions,
                   videorender::fixture3d::RenderUse::Export,
                   sceneInputs, output, error);
}

bool GlbNativeRenderSeam::publishSurfaceMaterial (
    const GlbNativeSceneAdmission& admission,
    const surfacematerialbinding::ImportedSceneMaterialRequest& request,
    std::string& error)
{
    auto reject = [&] (std::string diagnostic)
    {
        if (request.programRevision != 0
            && request.programRevision == highestMaterialRevisionSeen_
            && request.programRevision > lastGoodMaterialRevision_)
            rejectedMaterialRevision_ = request.programRevision;
        return fail (error, "native GLB material publication rejected",
                     std::move (diagnostic));
    };
    if (request.programRevision != 0
        && request.programRevision < highestMaterialRevisionSeen_)
        return reject ("surface material request revision is stale");
    if (request.programRevision > highestMaterialRevisionSeen_)
        highestMaterialRevisionSeen_ = request.programRevision;
    if (request.programRevision != 0
        && request.programRevision == rejectedMaterialRevision_)
        return reject ("surface material request revision was already rejected");
    if (! admission.valid())
        return reject ("scene admission is empty");

    using videohelper::materialprogram::BackendTarget;
    const auto backendName = backend_.info().backend;
    const auto target = backendName == "metal" ? BackendTarget::Metal
                      : backendName == "opengl" ? BackendTarget::OpenGl
                                                 : BackendTarget::Invalid;
    if (target == BackendTarget::Invalid)
        return reject ("native fixture backend has no admitted material compiler target");

    std::string diagnostic;
    auto candidate = videorender::fixture3d::admitSurfaceMaterialBinding (
        admission.scene, request, target, diagnostic);
    if (candidate == nullptr)
        return reject (std::move (diagnostic));

    const auto materialScene = materialScene_.lock();
    if (request.programRevision == lastGoodMaterialRevision_)
    {
        if (materialScene == admission.scene && lastGoodMaterial_ != nullptr
            && lastGoodDiffractionMaterial_ == nullptr
            && lastGoodMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodMaterial_->structuralRevision() == candidate->structuralRevision()
            && lastGoodMaterial_->evaluationRevision() == candidate->evaluationRevision()
            && lastGoodMaterial_->bindingDigest() == candidate->bindingDigest())
        {
            error.clear();
            return true;
        }
        const bool nextEvaluation = materialScene == admission.scene
            && lastGoodMaterial_ != nullptr
            && lastGoodMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodMaterial_->structuralRevision() == candidate->structuralRevision()
            && candidate->evaluationRevision() > lastGoodMaterial_->evaluationRevision();
        if (! nextEvaluation)
            return reject ("surface material revision already names a different immutable request");
    }
    if (lastGoodMaterial_ != nullptr && materialScene == admission.scene
        && lastGoodMaterial_->sceneRevision() != candidate->sceneRevision())
        return reject ("surface material request scene revision does not match the immutable admission");

    materialScene_ = admission.scene;
    lastGoodMaterial_ = std::move (candidate);
    lastSurfaceProgramsIdentity_.clear();
    lastGoodDiffractionMaterial_.reset();
    lastGoodMaterialRevision_ = request.programRevision;
    rejectedMaterialRevision_ = 0;
    error.clear();
    return true;
}

bool GlbNativeRenderSeam::publishSurfacePrograms (
    const GlbNativeSceneAdmission& admission,
    const geometrysurfacematerial::ObjectPrograms& programs,
    const videowire::geometry::RuntimeFieldEvaluation& evaluation, std::string& error,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest* diffraction,
    const visualimportedscenerender::GlassCausticSelection* caustic)
{
    if (!admission.valid() || programs.empty())
        return fail(error, "native Surface collection publication rejected", "scene or object programs are empty");
    const auto& request = programs.front().program.material;
    visualimportedscenerender::Request operation;
    operation.sourceStableId = admission.scene->id.value;
    operation.renderStableId = operation.sourceStableId == 1 ? 2 : 1;
    operation.sceneSnapshot = admission.scene;
    operation.structuralRevision = request.structuralRevision;
    operation.evaluationRevision = request.evaluationRevision;
    operation.surfacePrograms = programs;
    if (diffraction) operation.diffractionMaterial = *diffraction;
    if (caustic) operation.glassCaustic = *caustic;
    if (!visualimportedscenerender::valid(operation))
        return fail(error, "native Surface collection publication rejected", "mixed object identities or revisions are invalid");
    if (request.programRevision < highestMaterialRevisionSeen_)
        return fail(error, "native Surface collection publication rejected", "program revision is stale");
    highestMaterialRevisionSeen_ = std::max(highestMaterialRevisionSeen_, request.programRevision);
    const auto identity = videohelper::sha256Text(visualimportedscenerender::encode(operation));
    const bool sameScene = materialScene_.lock() == admission.scene;
    if (sameScene && lastGoodMaterial_ && request.programRevision == lastGoodMaterialRevision_
        && (request.evaluationRevision < lastGoodMaterial_->evaluationRevision()
            || (request.evaluationRevision == lastGoodMaterial_->evaluationRevision()
                && identity != lastSurfaceProgramsIdentity_)))
        return fail(error, "native Surface collection publication rejected", "revision names a different immutable object collection");
    using videohelper::materialprogram::BackendTarget;
    const auto backend = backend_.info().backend;
    const auto target = backend == "metal" ? BackendTarget::Metal
        : backend == "opengl" ? BackendTarget::OpenGl : BackendTarget::Invalid;
    auto candidate = videorender::fixture3d::admitSurfaceMaterialCollection(admission.scene, programs, evaluation, target, error);
    if (!candidate) return false;
    std::shared_ptr<const videorender::fixture3d::AdmittedDiffractionMaterialBinding> grating;
    if (diffraction)
    {
        const auto consumer = caustic ? arbitgpu::NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders
                                     : arbitgpu::NativeFixtureDiffractionConsumer::FiniteLobe;
        grating = videorender::fixture3d::admitDiffractionMaterialBinding(admission.scene, *diffraction, target, error, true, consumer);
        if (!grating) return false;
        if (caustic)
        {
            auto mixed = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*grating->nativeProgram());
            auto first = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*candidate->nativeProgram());
            first->objectPrograms.clear();
            mixed->objectPrograms.push_back(std::move(first));
            mixed->objectPrograms.insert(mixed->objectPrograms.end(),
                candidate->nativeProgram()->objectPrograms.begin(), candidate->nativeProgram()->objectPrograms.end());
            const arbitgpu::smoothglass::DistinctGratingAdmission gratingOwner {
                grating->object(), grating->materialIR() };
            arbitgpu::smoothglass::Prepared prepared;
            if (!arbitgpu::smoothglass::prepare(admission.scene, mixed, {}, prepared, error, {}, &gratingOwner)) return false;
            if (!prepared.enabled || prepared.object != caustic->glassObject || prepared.material != caustic->glassMaterial)
                return fail(error, "native discrete grating publication rejected", "selected closed glass owner is not retained");
        }
    }
    if (sameScene && lastGoodMaterial_ && identity == lastSurfaceProgramsIdentity_
        && candidate->bindingDigest() == lastGoodMaterial_->bindingDigest())
    { error.clear(); return true; }
    materialScene_ = admission.scene;
    lastGoodMaterial_ = std::move(candidate);
    lastGoodDiffractionMaterial_ = std::move(grating);
    lastSurfaceProgramsIdentity_ = identity;
    lastGoodMaterialRevision_ = request.programRevision;
    rejectedMaterialRevision_ = 0;
    error.clear(); return true;
}

void GlbNativeRenderSeam::clearSurfaceMaterial (
    const GlbNativeSceneAdmission& admission) noexcept
{
    if (materialScene_.lock() != admission.scene || lastGoodMaterial_ == nullptr)
        return;
    lastGoodMaterial_.reset();
    lastSurfaceProgramsIdentity_.clear();
    if (lastGoodDiffractionMaterial_ == nullptr)
    {
        materialScene_.reset();
        highestMaterialRevisionSeen_ = 0;
        lastGoodMaterialRevision_ = 0;
        rejectedMaterialRevision_ = 0;
    }
}

bool GlbNativeRenderSeam::publishDiffractionMaterial (
    const GlbNativeSceneAdmission& admission,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& request,
    std::string& error)
{
    auto reject = [&] (std::string diagnostic)
    {
        if (request.materialRevision != 0
            && request.materialRevision == highestMaterialRevisionSeen_
            && request.materialRevision > lastGoodMaterialRevision_)
            rejectedMaterialRevision_ = request.materialRevision;
        return fail (error, "native GLB diffraction material publication rejected",
                     std::move (diagnostic));
    };
    if (request.materialRevision != 0
        && request.materialRevision < highestMaterialRevisionSeen_)
        return reject ("diffraction material request revision is stale");
    if (request.materialRevision > highestMaterialRevisionSeen_)
        highestMaterialRevisionSeen_ = request.materialRevision;
    if (request.materialRevision != 0
        && request.materialRevision == rejectedMaterialRevision_)
        return reject ("diffraction material request revision was already rejected");
    if (! admission.valid())
        return reject ("scene admission is empty");

    using videohelper::materialprogram::BackendTarget;
    const auto backendName = backend_.info().backend;
    const auto target = backendName == "metal" ? BackendTarget::Metal
                      : backendName == "opengl" ? BackendTarget::OpenGl
                                                 : BackendTarget::Invalid;
    if (target == BackendTarget::Invalid)
        return reject ("native fixture backend has no admitted diffraction target");

    std::string diagnostic;
    auto candidate = videorender::fixture3d::admitDiffractionMaterialBinding (
        admission.scene, request, target, diagnostic);
    if (candidate == nullptr)
        return reject (std::move (diagnostic));

    const auto materialScene = materialScene_.lock();
    if (request.materialRevision == lastGoodMaterialRevision_)
    {
        if (materialScene == admission.scene && lastGoodMaterial_ == nullptr
            && lastGoodDiffractionMaterial_ != nullptr
            && lastGoodDiffractionMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodDiffractionMaterial_->structuralRevision()
                == candidate->structuralRevision()
            && lastGoodDiffractionMaterial_->evaluationRevision()
                == candidate->evaluationRevision()
            && lastGoodDiffractionMaterial_->bindingDigest() == candidate->bindingDigest())
        {
            error.clear();
            return true;
        }
        const bool nextEvaluation = materialScene == admission.scene
            && lastGoodMaterial_ == nullptr && lastGoodDiffractionMaterial_ != nullptr
            && lastGoodDiffractionMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodDiffractionMaterial_->structuralRevision()
                == candidate->structuralRevision()
            && candidate->evaluationRevision()
                > lastGoodDiffractionMaterial_->evaluationRevision();
        if (! nextEvaluation)
            return reject ("diffraction material revision already names a different immutable request");
    }
    if (lastGoodDiffractionMaterial_ != nullptr && materialScene == admission.scene
        && lastGoodDiffractionMaterial_->sceneRevision() != candidate->sceneRevision())
        return reject ("diffraction material scene revision does not match the immutable admission");

    materialScene_ = admission.scene;
    lastGoodMaterial_.reset();
    lastGoodDiffractionMaterial_ = std::move (candidate);
    lastGoodMaterialRevision_ = request.materialRevision;
    rejectedMaterialRevision_ = 0;
    error.clear();
    return true;
}

void GlbNativeRenderSeam::clearDiffractionMaterial (
    const GlbNativeSceneAdmission& admission) noexcept
{
    if (materialScene_.lock() != admission.scene
        || lastGoodDiffractionMaterial_ == nullptr)
        return;
    lastGoodDiffractionMaterial_.reset();
    if (lastGoodMaterial_ == nullptr)
    {
        materialScene_.reset();
        highestMaterialRevisionSeen_ = 0;
        lastGoodMaterialRevision_ = 0;
        rejectedMaterialRevision_ = 0;
    }
}

bool GlbNativeRenderSeam::render (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    videorender::fixture3d::RenderUse use,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    // Until the complete backend/caller/cache census and authenticated world
    // consumer exist, refuse before even caustic output retirement can run.
    if (sceneInputs.localSceneFrame || (sceneInputs.previousMotion && sceneInputs.previousMotion->localSceneFrame))
        return fail(error,"native GLB render rejected","static local backend retention and typed consumers are not admitted");
    if (sceneInputs.glassCaustic)
    {
        output = {};
        if (rejectedMaterialRevision_ > lastGoodMaterialRevision_)
            return fail(error, "native GLB render rejected", "current caustic material revision is not renderable");
    }
    if (! admission.valid())
        return fail (error, "native GLB render rejected", "scene admission is empty");
    if (use == videorender::fixture3d::RenderUse::Export
        && rejectedMaterialRevision_ > lastGoodMaterialRevision_)
        return fail (error, "native GLB render rejected",
                     "current material revision is not exportable");

    videorender::fixture3d::RenderedFrame frame;
    std::string diagnostic;
    const auto materialScene = materialScene_.lock();
    const auto material = materialScene == admission.scene
        ? lastGoodMaterial_ : nullptr;
    const auto diffractionMaterial = materialScene == admission.scene
        ? lastGoodDiffractionMaterial_ : nullptr;
    surfacematerial::MaterialEvaluationInputs materialInputs;
    materialInputs.timeSeconds = sceneInputs.timeSeconds;
    const auto rendered = material && diffractionMaterial
        ? renderer_.renderMixed(admission.scene, material, diffractionMaterial, sceneInputs,
                                dimensions, use, videorender::fixture3d::kNativeGpuCapability, frame, diagnostic)
        : diffractionMaterial != nullptr
        ? (use == videorender::fixture3d::RenderUse::Preview
            ? renderer_.renderPreview(admission.scene, diffractionMaterial, sceneInputs,
                                      dimensions, videorender::fixture3d::kNativeGpuCapability,
                                      frame, diagnostic)
            : renderer_.renderExport(admission.scene, diffractionMaterial, sceneInputs,
                                     dimensions, videorender::fixture3d::kNativeGpuCapability,
                                     frame, diagnostic))
        : (use == videorender::fixture3d::RenderUse::Preview
            ? renderer_.renderPreview(admission.scene, material, materialInputs, sceneInputs,
                                      dimensions, videorender::fixture3d::kNativeGpuCapability,
                                      frame, diagnostic)
            : renderer_.renderExport(admission.scene, material, materialInputs, sceneInputs,
                                     dimensions, videorender::fixture3d::kNativeGpuCapability,
                                     frame, diagnostic));
    if (! rendered)
        return fail (error, "native GLB render rejected", std::move (diagnostic));
    if (frame.pending)
    {
        // Pending is a retry state. It publishes no draws, materials, or textures.
        output = GlbNativeFrameReceipt {};
        output.rendered = std::move (frame);
        error.clear();
        return true;
    }

    GlbNativeFrameReceipt receipt;
    receipt.rendered = std::move (frame);
    receipt.draws.reserve (admission.scene->objectCount);
    for (std::size_t index = 0; index < admission.scene->objectCount; ++index)
    {
        const auto& object = admission.scene->objects[index];
        receipt.draws.push_back ({ object.id, object.parent, object.material, object.transform,
            object.firstVertex, object.vertexCount, object.firstIndex, object.indexCount });
    }
    receipt.materials.reserve (admission.scene->materialCount);
    for (std::size_t index = 0; index < admission.scene->materialCount; ++index)
    {
        const auto& materialRecord = admission.scene->materials[index];
        receipt.materials.push_back ({ materialRecord.id,
            materialRecord.baseColorTexture, materialRecord.metallicRoughnessTexture,
            materialRecord.normalTexture, materialRecord.occlusionTexture,
            materialRecord.emissiveTexture, materialRecord.baseColor,
            materialRecord.opacity, materialRecord.metallic, materialRecord.roughness,
            materialRecord.emissive, materialRecord.normalScale,
            materialRecord.occlusionStrength, materialRecord.alphaMode,
            materialRecord.alphaCutoff,
            materialRecord.doubleSided });
        const auto addRole = [&] (HarmonicMIDI::grid::SceneTextureId texture,
                                  GlbTextureRole role, GlbTextureColorSpace colorSpace)
        {
            if (texture.isValid())
                receipt.textureRoles.push_back ({ materialRecord.id, texture, role, colorSpace });
        };
        addRole (materialRecord.baseColorTexture, GlbTextureRole::BaseColor,
                 GlbTextureColorSpace::Srgb);
        addRole (materialRecord.metallicRoughnessTexture, GlbTextureRole::MetallicRoughness,
                 GlbTextureColorSpace::Linear);
        addRole (materialRecord.normalTexture, GlbTextureRole::Normal,
                 GlbTextureColorSpace::Linear);
        addRole (materialRecord.occlusionTexture, GlbTextureRole::Occlusion,
                 GlbTextureColorSpace::Linear);
        addRole (materialRecord.emissiveTexture, GlbTextureRole::Emissive,
                 GlbTextureColorSpace::Srgb);
    }
    receipt.textures.reserve (admission.scene->textureCount);
    for (std::size_t index = 0; index < admission.scene->textureCount; ++index)
    {
        const auto& texture = admission.scene->textures[index];
        receipt.textures.push_back ({ texture.id, texture.width, texture.height,
            texture.firstTexel, texture.width * texture.height,
            texture.magFilter, texture.minFilter, texture.wrapS, texture.wrapT });
    }
    receipt.activeCamera = admission.scene->activeCamera;
    receipt.cameras.assign (admission.scene->cameras.begin(),
                            admission.scene->cameras.begin() + admission.scene->cameraCount);
    receipt.lights.assign (admission.scene->lights.begin(),
                           admission.scene->lights.begin() + admission.scene->lightCount);
    if (receipt.rendered.nativeFrame != nullptr)
    {
        const auto descriptor = receipt.rendered.nativeFrame->colorTextureDescriptor();
        receipt.nativeBackend = receipt.rendered.nativeFrame->backend();
        receipt.nativeResourceCacheIdentity
            = receipt.rendered.nativeFrame->nativeResourceCacheIdentity();
        receipt.nativeDeviceOrContextIdentity = descriptor.deviceOrContextIdentity;
        receipt.nativeRendererGeneration = descriptor.rendererGeneration;
        receipt.downstreamHandoffComplete = descriptor.complete()
            && receipt.nativeResourceCacheIdentity != 0;
    }
    output = std::move (receipt);
    error.clear();
    return true;
}

} // namespace videohelper::gltf
