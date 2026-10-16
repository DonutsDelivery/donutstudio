#include "imported_animated_scene_payload_execution.h"
#include "imported_scene_payload_execution.h"
#include "imported_light_evaluation.h"
#include "geometry_score_field.h"
#include "../../shared/Visual3DSceneComposition.h"
#include "../../shared/GeometrySurfaceMaterial.h"
#include "loop_step_probe.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

namespace videohelper::modelpayload
{
namespace
{
template <typename T>
std::shared_ptr<const T> retainCompositionAllocation(std::shared_ptr<const T> value,
    const std::shared_ptr<const NativeCompositionRetentionBudget::Charge>& charge)
{
    struct Retained final { std::shared_ptr<const NativeCompositionRetentionBudget::Charge> charge; std::shared_ptr<const T> value; };
    auto retained = std::make_shared<Retained>(Retained {charge, value});
    return {std::move(retained), value.get()};
}

template <typename T>
std::shared_ptr<const T> retainSceneAnimationAllocation(std::shared_ptr<const T> value,
    const std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge>& charge)
{
    struct Retained final
    {
        std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> charge;
        std::shared_ptr<const T> value;
    };
    auto retained = std::make_shared<Retained>(Retained {charge, value});
    return {std::move(retained), value.get()};
}

// Every independently escaping parent alias keeps both logical storage and the
// permanent attempted-work ledger. It never owns the sidecar, avoiding cycles.
template <typename T>
std::shared_ptr<const T> retainParentAllocation(std::shared_ptr<const T> value,
    const std::shared_ptr<const NativeCompositionRetentionBudget::Charge>& charge,
    const std::shared_ptr<NativeSceneAnimationCompositionBudget>& budget)
{
    struct Retained final
    {
        std::shared_ptr<const NativeCompositionRetentionBudget::Charge> charge;
        std::shared_ptr<NativeSceneAnimationCompositionBudget> budget;
        std::shared_ptr<const T> value;
    };
    auto retained=std::make_shared<Retained>(Retained {charge,budget,value});
    return {std::move(retained),value.get()};
}

// The private owner ledger covers new IR/assets. Its retained original inputs
// are charged separately here, including all reachable vector/string capacities.
// Shared inputs can be counted more than once across owners; no allocation or
// reference-count elision is assumed, and no data is copied by this census.
bool compositionInputBytes(const visualmodelassetpayload::PayloadPtr& payload,
    const gltf::GlbStaticMeshDocument& document,const gltf::GlbAnimationDocument& animation,
    const HarmonicMIDI::grid::Visual3DScene& imported,const HarmonicMIDI::grid::Visual3DScene& procedural,
    const HarmonicMIDI::grid::Visual3DScene& composed,std::size_t& bytes,
    std::size_t& callerWork,std::size_t workLimit,std::string& error,
    const std::shared_ptr<NativeSceneAnimationCompositionBudget>& animationBudget = {})
{
    bytes=sizeof(*payload)+sizeof(document)+sizeof(animation)+3*sizeof(imported);
    if (bytes>(workLimit-callerWork)/2)
    {error="native composition retained input census exceeds named caller work";return false;}
    callerWork+=2*bytes;
    if (animationBudget && !animationBudget->work(bytes,2,error)) return false;
    const auto charge=[&](std::size_t count,std::size_t stride) {
        if (bytes>NativeCompositionRetentionBudget::kMaximumBytes
            || count>(NativeCompositionRetentionBudget::kMaximumBytes-bytes)/stride)
        {error="native composition retained original input capacities exceed logical CPU capacity";return false;}
        const auto lanes=count*stride;
        if (lanes>(workLimit-callerWork)/2)
        {error="native composition retained input census exceeds named caller work";return false;}
        // Capacity/header work is debited before any following nested visit.
        if (animationBudget && !animationBudget->work(lanes,2,error)) return false;
        callerWork+=2*lanes;bytes+=lanes;return true;
    };
    const auto vector=[&](const auto& values) {
        using Value=typename std::decay_t<decltype(values)>::value_type;
        return charge(values.capacity(),sizeof(Value));
    };
    const auto text=[&](const std::string& value) {
        if (value.capacity()==std::numeric_limits<std::size_t>::max())
        {error="native composition retained original identity capacity overflows";return false;}
        return charge(value.capacity()+1,1);
    };
    const auto metadata=[&](const gltf::GlbMetadata& value) {
        if (!text(value.generator) || !vector(value.extensionsUsed) || !vector(value.extensionsRequired)) return false;
        for (const auto& item:value.extensionsUsed) if (!text(item)) return false;
        for (const auto& item:value.extensionsRequired) if (!text(item)) return false;
        return true;
    };
    const auto& key=payload->key();
    if (!vector(payload->bytes()) || !text(key.id) || !text(key.contentSha256) || !text(key.sourceMediaType)
        || !metadata(document.metadata) || !metadata(animation.metadata)
        || !vector(document.bufferViews) || !vector(document.accessors) || !vector(document.samplers)
        || !vector(document.images) || !vector(document.textures) || !vector(document.materials)
        || !vector(document.meshes) || !vector(document.cameras) || !vector(document.lights)
        || !vector(document.nodes) || !vector(document.scenes)
        || !vector(imported.textureTexels) || !vector(procedural.textureTexels) || !vector(composed.textureTexels)) return false;
    // Full Visual3DScene copies retain every fixed light slot. Validation only
    // visits lightCount, so inactive identities need the same census debit.
    if (animationBudget)
        for (const auto* scene:{&imported,&procedural,&composed})
            for (const auto& light:scene->lights)
                if (!text(light.environmentImage.asset.id) || !text(light.environmentImage.asset.contentSha256)) return false;
    for (const auto& value:document.samplers) if (!text(value.name)) return false;
    for (const auto& value:document.images)
        if (!text(value.name) || !text(value.mimeType) || !vector(value.decodedRgba8)) return false;
    for (const auto& value:document.textures) if (!text(value.name)) return false;
    for (const auto& value:document.materials) if (!text(value.name)) return false;
    for (const auto& value:document.cameras) if (!text(value.name)) return false;
    for (const auto& value:document.lights) if (!text(value.name)) return false;
    for (const auto& value:document.nodes) if (!text(value.name) || !vector(value.children)) return false;
    for (const auto& value:document.scenes) if (!text(value.name) || !vector(value.rootNodes)) return false;
    for (const auto& mesh:document.meshes)
    {
        if (!text(mesh.name) || !vector(mesh.primitives)) return false;
        for (const auto& value:mesh.primitives)
            if (!vector(value.positions) || !vector(value.normals) || !vector(value.tangents)
                || !vector(value.texCoords0) || !vector(value.colors0) || !vector(value.indices)
                || !vector(value.cornerSources)) return false;
    }
    if (!vector(animation.clips) || !vector(animation.renderBindings)
        || !vector(animation.nodeMeshes) || !vector(animation.sceneMeshes)) return false;
    for (const auto& value:animation.renderBindings)
        if (!vector(value.jointBaseTransforms) || !vector(value.morphBaseWeights)) return false;
    for (const auto& named:animation.clips)
    {
        if (!text(named.name) || !vector(named.jointBindings) || !vector(named.morphBindings)
            || !vector(named.sceneTransformTargets)) return false;
        for (const auto& value:named.morphBindings) if (!vector(value.targets)) return false;
        if (named.clip)
        {
            if (!charge(1,sizeof(*named.clip)) || !vector(named.clip->tracks())) return false;
            for (const auto& track:named.clip->tracks())
                if (!vector(track.keyTimesSeconds()) || !vector(track.values())) return false;
        }
    }
    if (animation.deformation)
    {
        const auto& asset=*animation.deformation;
        if (!charge(1,sizeof(asset)) || !vector(asset.skins()) || !vector(asset.meshes())) return false;
        for (const auto& skin:asset.skins())
            if (!vector(skin.joints()) || !vector(skin.inverseBindMatrixValues())) return false;
        for (const auto& mesh:asset.meshes())
        {
            if (!vector(mesh.jointWeightSets()) || !vector(mesh.morphTargets())) return false;
            for (const auto& influence:mesh.jointWeightSets())
                if (!vector(influence.jointIndices()) || !vector(influence.weights())) return false;
            for (const auto& target:mesh.morphTargets())
                if (!vector(target.positionDeltas()) || !vector(target.normalDeltas()) || !vector(target.tangentDeltas())) return false;
        }
    }
    return true;
}

bool compositionFrameBytes(const arbitgpu::NativeDeformationCompositionOwner& owner,
    const ImportedAnimationDeformationEvaluation& evaluation,
    std::size_t& bytes, std::string& error)
{
    if (!evaluation.deformation) { error="native composition requires an actual deformation snapshot"; return false; }
    using namespace arbitgpu;
    using Matrix=std::array<float,16>;
    const auto nodes=owner.importedNodeCount();
    // Mirrors the private frame's admitted logical storage, plus both caller
    // world tables, typed preflight scratch and alias wrappers. Count capacity,
    // never assume a move/return elides an allocation or simultaneous owner.
    bytes=2*NativeDeformationCompositionFrame::singleLeafBytes()+nodes*(3*sizeof(Matrix)+4*sizeof(std::uint8_t))
        +2*64*sizeof(Matrix)+2*64*256+2*64*sizeof(NativeDeformationFrameData)
        +2*nodes*sizeof(NativeSceneMotionSample::ImportedNodeWorldMatrix)
        +64*(64*sizeof(NativeDeformationJointBaseTransform)+4*sizeof(Matrix)+sizeof(std::uint8_t))
        +sizeof(*evaluation.deformation)+18*sizeof(float)
        +4*nodes*sizeof(NativeSceneMotionSample::ImportedNodeWorldMatrix)
        +2*nodes*(sizeof(bool)+sizeof(const NativeSceneMotionSample::ImportedNodeWorldMatrix*))
        +4*sizeof(NativeDeformationFrameData)+12*sizeof(std::shared_ptr<const void>)
        +sizeof(NativeCompositionRetentionBudget::Charge);
    const auto charge=[&](std::size_t n,std::size_t stride) {
        if (bytes>NativeCompositionRetentionBudget::kMaximumBytes
            || n>(NativeCompositionRetentionBudget::kMaximumBytes-bytes)/stride)
        {error="native composition caller snapshot storage exceeds logical capacity";return false;}
        bytes+=n*stride;return true;
    };
    const auto& snapshot=*evaluation.deformation;
    if (!charge(snapshot.jointTransforms().capacity(),sizeof(visualdeformation::JointTransformEvaluation))
        || !charge(snapshot.morphWeights().capacity(),sizeof(visualdeformation::MorphWeightEvaluation)))
    {error="native composition caller snapshot capacity overflows its logical schedule";return false;}
    for (const auto& morph:snapshot.morphWeights())
        if (!charge(1,sizeof(visualdeformation::MorphWeightSample)) || !charge(morph.values().capacity(),sizeof(float))
            || !charge(morph.targets().capacity(),sizeof(visualdeformation::MorphTargetId)))
        {error="native composition caller morph capacity overflows its logical schedule";return false;}
    if (evaluation.sceneAnimation)
    {
        if (!charge(1,sizeof(*evaluation.sceneAnimation))
            || !charge(evaluation.sceneAnimation->tracks().capacity(),sizeof(visualanimation::TrackSample))) return false;
        for (const auto& track:evaluation.sceneAnimation->tracks())
            if (!charge(track.values().capacity(),sizeof(float))) return false;
    }
    return true;
}

bool reject(std::string& error, const char* message)
{
    error = message;
    return false;
}

std::shared_ptr<const arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices>
importedNodeWorlds(const gltf::GlbStaticMeshDocument& base,
    const arbitgpu::NativeDeformationScene& source,
    const HarmonicMIDI::grid::Visual3DScene& imported,
    const ImportedAnimationDeformationEvaluation& evaluation,
    const std::shared_ptr<const arbitgpu::NativeDeformationCompositionFrame>& composition = {},
    std::string* failure = nullptr)
{
    using Sample = arbitgpu::NativeSceneMotionSample;
    if (!source.scene) return {};
    std::vector<bool> required(base.nodes.size(), false);
    for (std::size_t index = 0; index < imported.objectCount; ++index)
    {
        const auto id = imported.objects[index].id.value;
        if (id == 0) return {};
        std::optional<std::size_t> node = (id - 1u) / 65536u;
        for (std::size_t visits = 0; node; ++visits)
        {
            if (*node >= base.nodes.size() || visits >= base.nodes.size()) return {};
            required[*node] = true;
            node = base.nodes[*node].parent;
        }
    }
    const auto count = static_cast<std::size_t>(std::count(required.begin(), required.end(), true));
    const auto bytes = sizeof(Sample::ImportedNodeWorldMatrices)
        + count * sizeof(Sample::ImportedNodeWorldMatrix);
    if (!arbitgpu::sceneMotionSnapshotFits(evaluation.deformation, bytes)) return {};

    Sample::ImportedNodeWorldMatrices sampled;
    if (evaluation.deformation)
    {
        arbitgpu::NativeDeformationFrameData frame;
        std::string diagnostic;
        // Reuse the draw's evaluator. Reconstructing worlds from palettes would
        // lose precision through inverse object and inverse bind transforms.
        if (!arbitgpu::prepareNativeDeformationFrame(source, *evaluation.deformation,
                composition, frame, diagnostic, &sampled))
        {
            if (failure) *failure = diagnostic;
            return {};
        }
    }
    std::vector<const Sample::ImportedNodeWorldMatrix*> byNode(base.nodes.size(), nullptr);
    for (const auto& entry : sampled)
    {
        if (entry.nodeStableId == 0 || entry.nodeStableId > base.nodes.size()) return {};
        byNode[entry.nodeStableId - 1u] = &entry;
    }
    const auto captured = [&](std::uint64_t id) {
        return id > 0 && id <= required.size()
            && (!required[id - 1u] || byNode[id - 1u] != nullptr);
    };
    if (evaluation.deformation)
        for (const auto& transform : evaluation.deformation->jointTransforms())
            if (!captured(transform.animationTarget().value)) return {};
    if (evaluation.sceneAnimation)
        for (const auto& track : evaluation.sceneAnimation->tracks())
            if (track.channel() != visualanimation::Channel::MorphWeights
                && !captured(track.target().value)) return {};

    auto retained = std::make_shared<Sample::ImportedNodeWorldMatrices>();
    retained->reserve(count);
    for (std::size_t index = 0; index < required.size(); ++index)
    {
        if (!required[index]) continue;
        const auto& world = byNode[index] ? byNode[index]->world : base.nodes[index].worldTransform;
        if (!std::all_of(world.begin(), world.end(), [](float value) { return std::isfinite(value); }))
            return {};
        retained->push_back({index + 1u, world});
    }
    if (!arbitgpu::sceneMotionSnapshotFits(evaluation.deformation,
            sizeof(*retained) + retained->capacity() * sizeof(Sample::ImportedNodeWorldMatrix))) return {};
    return retained;
}

// Keeps the render bindings whose node draws in the adapted leaf scene. Each
// kept node's primitives must be contiguous and match its deformation mesh.
bool collectImportedLeafBindings(const gltf::GlbAnimationDocument& document,
    const HarmonicMIDI::grid::Visual3DScene& adapted, visualanimationimport::StableId meshStableId,
    std::vector<gltf::GlbDeformationRenderBinding>& bindings, std::string& error)
{
    const auto end = adapted.objects.begin() + adapted.objectCount;
    for (const auto& binding : document.renderBindings)
    {
        if (meshStableId != 0 && binding.mesh.value != meshStableId) continue;
        const auto expectedObject = static_cast<std::uint64_t>(binding.nodeIndex) * 65536u + 1u;
        const auto object = std::find_if(adapted.objects.begin(), end,
            [expectedObject](const auto& candidate) { return candidate.id.value == expectedObject; });
        if (object == end) continue;
        const auto* mesh = document.deformation->findMesh(binding.mesh);
        if (!mesh || mesh->vertexCount() > adapted.vertexCount - object->firstVertex)
            return reject(error, "imported animated mesh has incompatible or changing topology");
        std::size_t vertices = 0;
        for (auto part = object; part != end && (part->id.value - 1u) / 65536u == binding.nodeIndex; ++part)
        {
            if (part->firstVertex != object->firstVertex + vertices)
                return reject(error, "imported animated mesh primitives are not contiguous");
            vertices += part->vertexCount;
        }
        if (vertices != mesh->vertexCount())
            return reject(error, "imported animated mesh primitive topology does not match its deformation");
        bindings.push_back(binding);
    }
    return true;
}

// Every drawn node whose mesh deforms needs its own exact binding.
bool importedLeafDeformedNodesAreBound(const gltf::GlbStaticMeshDocument& decoded,
    const gltf::GlbAnimationDocument& document, const HarmonicMIDI::grid::Visual3DScene& adapted,
    const std::vector<gltf::GlbDeformationRenderBinding>& bindings, std::string& error)
{
    for (std::size_t index = 0; index < adapted.objectCount; ++index)
    {
        const auto nodeIndex = (adapted.objects[index].id.value - 1u) / 65536u;
        const auto meshIndex = decoded.nodes[nodeIndex].mesh;
        if (meshIndex && document.deformation && document.deformation->findMesh({*meshIndex + 1u})
            && std::none_of(bindings.begin(), bindings.end(),
                [nodeIndex](const auto& binding) { return binding.nodeIndex == nodeIndex; }))
            return reject(error, "animated mesh instances require a distinct mesh per scene node; this node has no exact deformation binding");
    }
    return true;
}

// Leaf k's raw object (node r, primitive p) becomes (k * stride + r, p). Leaf 0
// keeps its raw ID. The plugin lowers leaf material targets through the same
// shared mapping, so both sides name one composed object.
std::optional<HarmonicMIDI::grid::SceneObjectId> composedLeafObjectId(std::size_t leaf, std::size_t leafCount,
    HarmonicMIDI::grid::SceneObjectId raw) noexcept
{
    return HarmonicMIDI::grid::scenecomposition::composedImportedLeafObjectId(leaf, leafCount, raw);
}

// Moves every object, material and texture identity of one leaf scene into
// leaf k's range, including parent, material and texture references.
bool remapImportedLeafScene(HarmonicMIDI::grid::Visual3DScene& scene, std::size_t leaf, std::size_t leafCount)
{
    using HarmonicMIDI::grid::scenecomposition::encodeImportedLeafId;
    const auto remap = [&](std::uint32_t& value) {
        if (value == 0) return true;
        const auto encoded = encodeImportedLeafId(leaf, leafCount, value);
        if (!encoded) return false;
        value = *encoded;
        return true;
    };
    if (scene.textureCount > scene.textures.size() || scene.materialCount > scene.materials.size()
        || scene.objectCount > scene.objects.size()) return false;
    for (std::size_t i = 0; i < scene.textureCount; ++i)
        if (!remap(scene.textures[i].id.value)) return false;
    for (std::size_t i = 0; i < scene.materialCount; ++i)
    {
        auto& material = scene.materials[i];
        if (!remap(material.id.value) || !remap(material.baseColorTexture.value)
            || !remap(material.metallicRoughnessTexture.value) || !remap(material.normalTexture.value)
            || !remap(material.occlusionTexture.value) || !remap(material.emissiveTexture.value)) return false;
    }
    for (std::size_t i = 0; i < scene.objectCount; ++i)
    {
        auto& object = scene.objects[i];
        const auto id = composedLeafObjectId(leaf, leafCount, object.id);
        if (!id || !remap(object.material.value)) return false;
        object.id = *id;
        if (object.parent.isValid())
        {
            const auto parent = composedLeafObjectId(leaf, leafCount, object.parent);
            if (!parent) return false;
            object.parent = *parent;
        }
    }
    return true;
}

// Joins every leaf rest scene, leaf k in its own ID range, then the procedural
// snapshot, exactly as one leaf joins it. Leaf 0 keeps its raw IDs. Leaf 1+
// adopt leaf 0's cameras, lights and ambient color, which the composed scene
// replaces with the procedural records anyway.
std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composeImportedLeafScenes(
    const HarmonicMIDI::grid::Visual3DScene& procedural,
    const std::vector<std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>>& leaves, std::string& error)
{
    using HarmonicMIDI::grid::Visual3DScene;
    if (leaves.size() < 2 || leaves.size() > HarmonicMIDI::grid::scenecomposition::kMaximumImportedLeaves
        || std::any_of(leaves.begin(), leaves.end(), [](const auto& leaf) { return !leaf; }))
    {
        error = "Scene Compose refused: imported leaf scenes are missing";
        return {};
    }
    auto imported = std::make_shared<Visual3DScene>(*leaves[0]);
    for (std::size_t leaf = 1; leaf < leaves.size(); ++leaf)
    {
        auto copy = std::make_unique<Visual3DScene>(*leaves[leaf]);
        if (!remapImportedLeafScene(*copy, leaf, leaves.size()))
        {
            error = "Scene Compose refused: imported leaf " + std::to_string(leaf)
                + " has an object, material or texture identity outside its leaf range";
            return {};
        }
        copy->ambientColor = imported->ambientColor;
        copy->cameras = imported->cameras; copy->cameraCount = imported->cameraCount;
        copy->activeCamera = imported->activeCamera;
        copy->lights = imported->lights; copy->lightCount = imported->lightCount;
        std::string appendError;
        if (!HarmonicMIDI::grid::scenecomposition::appendScene(*copy, *imported, false, appendError))
        {
            error = "Scene Compose refused: imported leaf " + std::to_string(leaf) + " does not fit: " + appendError;
            return {};
        }
    }
    if (!HarmonicMIDI::grid::validateVisual3DScene(*imported).valid())
    {
        error = "Scene Compose refused: the joined imported leaves are not a valid scene";
        return {};
    }
    return gltf::composeImportedVisual3DScene(procedural, *imported, error);
}

// Logical storage of one further leaf sample in a multi-leaf frame: its
// snapshot, morph schedules and scene sample, with the same capacity rules
// as compositionFrameBytes.
bool compositionLeafSampleBytes(const ImportedAnimationDeformationEvaluation& evaluation,
    std::size_t& bytes, std::string& error)
{
    if (!evaluation.deformation) { error="native composition requires an actual deformation snapshot"; return false; }
    const auto charge=[&](std::size_t n,std::size_t stride) {
        if (bytes>NativeCompositionRetentionBudget::kMaximumBytes
            || n>(NativeCompositionRetentionBudget::kMaximumBytes-bytes)/stride)
        {error="native composition caller snapshot storage exceeds logical capacity";return false;}
        bytes+=n*stride;return true;
    };
    const auto& snapshot=*evaluation.deformation;
    if (!charge(1,sizeof(snapshot)+4*sizeof(std::shared_ptr<const void>))
        || !charge(snapshot.jointTransforms().capacity(),sizeof(visualdeformation::JointTransformEvaluation))
        || !charge(snapshot.morphWeights().capacity(),sizeof(visualdeformation::MorphWeightEvaluation))) return false;
    for (const auto& morph:snapshot.morphWeights())
        if (!charge(1,sizeof(visualdeformation::MorphWeightSample)) || !charge(morph.values().capacity(),sizeof(float))
            || !charge(morph.targets().capacity(),sizeof(visualdeformation::MorphTargetId))) return false;
    if (evaluation.sceneAnimation)
    {
        if (!charge(1,sizeof(*evaluation.sceneAnimation))
            || !charge(evaluation.sceneAnimation->tracks().capacity(),sizeof(visualanimation::TrackSample))) return false;
        for (const auto& track:evaluation.sceneAnimation->tracks())
            if (!charge(track.values().capacity(),sizeof(float))) return false;
    }
    return true;
}

// The moved node and every node below it. Null when the parent chain is not a
// finite walk to a root.
std::optional<bool> localTransformAffects(const gltf::GlbStaticMeshDocument& base,
    std::size_t node, std::uint64_t moved)
{
    for (std::size_t depth = 0; depth <= base.nodes.size(); ++depth)
    {
        if (node >= base.nodes.size()) return std::nullopt;
        if (node + 1u == moved) return true;
        if (!base.nodes[node].parent) return false;
        node = *base.nodes[node].parent;
    }
    return std::nullopt;
}

// Names why one object-local transform is not an exact rigid move of the admitted
// leaf, or returns null. Each affected drawn object needs the synthetic rigid skin
// of its node: no authored skin, joint 0 is that node, identity inverse bind, and no
// morph targets. An unaffected drawn object may keep its skin only while no joint of
// that skin lies in the moved subtree. The moved node must be drawn, and every drawn
// object unparented.
const char* localTransformRefusal(const gltf::GlbStaticMeshDocument& base,
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const std::vector<gltf::GlbDeformationRenderBinding>& bindings,
    const visualdeformation::DeformationAsset& deformation,
    const visualimportedobjectlocal::ObjectLocalTransformV1& local)
{
    constexpr const char* rigid = "Imported object local transforms require one rigid imported deformation leaf";
    const auto moved = static_cast<std::uint64_t>(local.nodeStableId);
    if (moved == 0 || moved > base.nodes.size())
        return "Imported object local transform does not name a node of the selected leaf";
    if ((base.nodes[moved - 1u].mesh ? *base.nodes[moved - 1u].mesh + 1u : 0u) != local.meshStableId)
        return "Imported object local transform does not name the selected mesh";
    bool movedDrawn = false;
    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        const auto& object = scene.objects[index];
        if (!object.indexCount) continue;
        if (!object.id.isValid()) return rigid;
        if (object.parent.isValid())
            return "Imported object local transforms with imported parenting are not rendered yet";
        const auto node = static_cast<std::size_t>((object.id.value - 1u) / 65536u);
        const auto affects = node < base.nodes.size() ? localTransformAffects(base, node, moved) : std::nullopt;
        if (!affects) return rigid;
        if (!*affects)
        {
            // An unaffected object keeps its authored chain. A skin joint inside the
            // moved subtree would move without the offset, so such a skin is refused.
            for (const auto& candidate : bindings)
            {
                if (candidate.nodeIndex != node || !candidate.skin.isValid()) continue;
                const auto* skin = deformation.findSkin(candidate.skin);
                if (!skin) return rigid;
                for (const auto& joint : skin->joints())
                {
                    if (!joint.id().isValid() || joint.id().value > base.nodes.size()) return rigid;
                    const auto jointNode = static_cast<std::size_t>(joint.id().value - 1u);
                    const auto joins = localTransformAffects(base, jointNode, moved);
                    if (!joins) return rigid;
                    if (*joins)
                        return "Imported object local transforms with skin joints in the moved subtree are not rendered yet";
                }
            }
            continue;
        }
        movedDrawn = movedDrawn || node + 1u == moved;
        const gltf::GlbDeformationRenderBinding* binding = nullptr;
        std::size_t bound = 0;
        for (const auto& candidate : bindings)
            if (candidate.nodeIndex == node) { binding = &candidate; ++bound; }
        if (bound != 1) return rigid;
        if (binding->authoredJointCount != 0)
            return "Imported object local transforms with skinned meshes are not rendered yet";
        const auto* mesh = deformation.findMesh(binding->mesh);
        if (!mesh) return rigid;
        if (!mesh->morphTargets().empty())
            return "Imported object local transforms with morph targets are not rendered yet";
        if (!binding->skin.isValid() || (mesh->skin().isValid() && mesh->skin() != binding->skin))
            return rigid;
        const auto* skin = deformation.findSkin(binding->skin);
        if (!skin || skin->joints().empty() || skin->joints().front().id().value != node + 1u) return rigid;
        constexpr std::array<float, 16> identity {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const auto& inverse = skin->inverseBindMatrixValues();
        if (inverse.size() < identity.size() || !std::equal(identity.begin(), identity.end(), inverse.begin()))
            return rigid;
    }
    return movedDrawn ? nullptr : "Imported object local transform names no drawn object of the selected leaf";
}

} // namespace

ImportedAnimatedScenePayloadExecution::ImportedAnimatedScenePayloadExecution(
    Store& store, arbitgpu::NativeDeformationBackend& backend,
    arbitgpu::NativeFixtureSceneBackend& sceneBackend) noexcept
    : store_(store), renderer_(backend), sceneRenderer_(sceneBackend)
{
}

bool ImportedAnimatedScenePayloadExecution::inspectCompatibility(
    const visualmodelassetpayload::PayloadPtr& payload,
    std::optional<std::size_t> sceneIndex,
    ImportedAnimatedSceneCompatibility& output,
    std::string& error)
{
    output = {};
    if (!payload)
        return reject(error, "imported animated scene compatibility payload is unavailable");

    const auto& bytes = payload->bytes();
    auto animationOptions = nativeImportedAnimationDecodeOptions(bytes.size(), sceneIndex);
    auto decodedScene = gltf::decodeAnimatedGlbBaseScene(
        bytes, animationOptions.admission, error);
    if (!decodedScene)
    {
        error = "imported scene base geometry rejected: " + error;
        return false;
    }
    if (decodedScene->metadata.animations == 0)
    {
        if (!gltf::adaptStaticGlbToVisual3DScene(*decodedScene, error))
        {
            error = "imported scene native adaptation rejected: " + error;
            return false;
        }
        error.clear();
        return true;
    }

    auto document = gltf::decodeGlbAnimations(bytes, animationOptions, error);
    if (!document)
    {
        error = "imported animation/deformation decode rejected: " + error;
        return false;
    }
    ImportedAnimatedSceneCompatibility inspected;
    inspected.clips.reserve(document->clips.size());
    std::map<visualanimationimport::StableId, bool> nativeMeshAdmission;
    for (std::size_t clipIndex = 0; clipIndex < document->clips.size(); ++clipIndex)
    {
        const auto& clip = document->clips[clipIndex];
        ImportedAnimationClipCompatibility candidate;
        candidate.stableId = static_cast<visualanimationimport::StableId>(clipIndex + 1u);
        candidate.name = clip.name;
        for (const auto meshStableId
             : importedAnimationCompatibleMeshStableIds(*document, clip))
        {
            if (!clip.sceneTransformTargets.empty())
            {
                if (gltf::adaptAnimatedGlbMeshToVisual3DSceneHeap(*decodedScene, error,
                        static_cast<std::size_t>(meshStableId - 1u), true))
                    candidate.compatibleMeshStableIds.push_back(meshStableId);
                error.clear();
                continue;
            }
            auto admitted = nativeMeshAdmission.find(meshStableId);
            if (admitted == nativeMeshAdmission.end())
            {
                const auto binding = std::find_if(
                    document->renderBindings.begin(), document->renderBindings.end(),
                    [meshStableId] (const auto& entry)
                    { return entry.mesh.value == meshStableId; });
                const auto deformationMesh = binding == document->renderBindings.end()
                    ? document->deformation->meshes().end()
                    : std::find_if(
                        document->deformation->meshes().begin(),
                        document->deformation->meshes().end(),
                        [&binding] (const auto& entry) { return entry.id() == binding->mesh; });
                const auto selectedMeshIndex = static_cast<std::size_t>(meshStableId - 1u);
                auto adapted = gltf::adaptAnimatedGlbMeshToVisual3DSceneHeap(
                    *decodedScene, error, selectedMeshIndex, true);
                const auto expectedObject = binding == document->renderBindings.end()
                    ? 0u : static_cast<std::uint64_t>(binding->nodeIndex) * 65536u + 1u;
                const auto nativeAdmitted = adapted && binding != document->renderBindings.end()
                    && deformationMesh != document->deformation->meshes().end()
                    && std::any_of(adapted->objects.begin(), adapted->objects.begin() + adapted->objectCount,
                        [expectedObject](const auto& object) { return object.id.value == expectedObject; });
                admitted = nativeMeshAdmission.emplace(meshStableId, nativeAdmitted).first;
                error.clear();
            }
            if (admitted->second)
                candidate.compatibleMeshStableIds.push_back(meshStableId);
        }
        inspected.clips.push_back(std::move(candidate));
    }
    output = std::move(inspected);
    error.clear();
    return true;
}

bool ImportedAnimatedScenePayloadExecution::executePreview(
    const ImportedAnimatedSceneRequest& request,
    ImportedAnimatedSceneReceipt& output,
    std::string& error)
{
    return execute(ImportedAnimationEvaluationOwner::Preview, request, output, error);
}

bool ImportedAnimatedScenePayloadExecution::executeExport(
    const ImportedAnimatedSceneRequest& request,
    ImportedAnimatedSceneReceipt& output,
    std::string& error)
{
    return execute(ImportedAnimationEvaluationOwner::Export, request, output, error);
}

bool ImportedAnimatedScenePayloadExecution::executeSceneAnimationComposition(
    ImportedAnimationEvaluationOwner use, const ImportedAnimatedSceneRequest& request,
    ImportedAnimationDeformationEvaluation evaluation, bool usedLastGoodMaterial,
    std::string materialDiagnostic, ImportedAnimatedSceneReceipt& output, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    const auto document=consumer_.admittedDocument();
    if (!sceneAnimationAdmissionCharge_ || !baseScene_ || !document || !evaluation.sceneAnimation
        || evaluation.deformation || !bindings_.empty())
        return reject(error,"camera/light composition has no exact no-deformation admission");
    if (!request.projectGeneration || !request.helperGeneration || request.renderClipId<=0)
        return reject(error,"camera/light composition requires exact project, helper and render clip ownership");
    if (request.runtimeInputs.compositionFrame || request.runtimeInputs.previousMotion
        || request.runtimeInputs.animatedCamera)
        return reject(error,"camera/light composition cannot accept caller-supplied animation authorities");
    // The sampler owns imported light records. Caller modulation rides lightOverrides,
    // applied after sampling. Caller records beyond the imported prefix are kept.
    const auto& callerLights=request.runtimeInputs.animatedLights;
    if (!callerLights.empty() && callerLights.size()<importedScene_->lightCount)
        return reject(error,"camera/light composition light records must cover every imported light");
    if (request.runtimeInputs.lightOverride && !request.importedLights.intensityMultipliers.empty())
        return reject(error,"Render 3D Light input replaces imported lighting");
    const auto nodes=baseScene_->nodes.size(), tracks=evaluation.sceneAnimation->tracks().size();
    const auto records=importedScene_->cameraCount+importedScene_->lightCount;
    std::size_t bytes=2*sizeof(NativeSceneAnimationCompositionFrame)+sizeof(NativeSceneAnimationCompositionOwner)
        +sizeof(arbitgpu::NativeDeformationScene)+2*sizeof(arbitgpu::NativeImportedSceneRuntimeInputs)
        +sizeof(videorender::fixture3d::RenderedFrame)+sizeof(ImportedAnimatedSceneReceipt)
        +sizeof(NativeSceneAnimationCompositionBudget::Charge)+32*sizeof(std::shared_ptr<const void>);
    const auto add=[&](std::size_t count,std::size_t stride) {
        if (bytes>NativeSceneAnimationCompositionBudget::kMaximumBytes
            || count>(NativeSceneAnimationCompositionBudget::kMaximumBytes-bytes)/stride)
            return reject(error,"camera/light composition frame exceeds logical CPU capacity");
        bytes+=count*stride;return true;
    };
    const auto& operation=request.operation;
    // Sampling owns local/world matrices, ancestry/state tables and records.
    // Frame publication, light controls, runtime copies and wrappers have their
    // own allowance. No deformation frame schedule covers these allocations.
    if (!add(nodes,4*sizeof(std::array<float,16>)+5*sizeof(std::uint8_t))
        || !add(4*Visual3DScene::kMaxLights,sizeof(SceneLightRecord))
        || !add(3*Visual3DScene::kMaxCameras,sizeof(SceneCameraRecord))
        || !add(request.importedLights.intensityMultipliers.size()+Visual3DScene::kMaxLights,
            2*(sizeof(std::pair<const std::uint32_t,float>)+4*sizeof(void*)))
        || !add(request.runtimeInputs.materialFrameTextures.size(),
            2*(sizeof(decltype(request.runtimeInputs.materialFrameTextures)::value_type)+4*sizeof(void*)))
        || !add(request.runtimeInputs.diffractionParameters.size(),
            2*(sizeof(decltype(request.runtimeInputs.diffractionParameters)::value_type)+4*sizeof(void*)))
        || !add(operation.asset.id.size()+operation.asset.contentSha256.size()+operation.asset.sourceMediaType.size()
            +operation.clipName.size()+operation.crossfadeClipName.size()+materialDiagnostic.size()+6,4)
        || !sceneAnimationBudget_->work(bytes,4,error)
        || !sceneAnimationBudget_->work(nodes,256,error)
        || !sceneAnimationBudget_->work(records*nodes,64,error)
        || !sceneAnimationBudget_->work(importedScene_->objectCount*nodes,32,error)
        || !sceneAnimationBudget_->work(tracks*(nodes+1),256,error)
        || !sceneAnimationBudget_->work(document->clips.capacity(),256,error)
        || !sceneAnimationBudget_->work(request.importedLights.intensityMultipliers.size(),128,error)) return false;
    if (!sceneAnimationBudget_->work(request.runtimeInputs.diffractionParameters.size(),64,error)) return false;
    for (const auto& parameter:request.runtimeInputs.diffractionParameters)
        if (!add(parameter.first.capacity()+1,2) || !sceneAnimationBudget_->work(parameter.first.capacity()+1,4,error)) return false;
    // Runtime override crosses the local runtime, renderer entry, renderer
    // body, backend entry and immutable frame. Sampled/imported/procedural
    // lights also cross vector growth, controls scratch and those publication
    // copies. Eight copies cover these simultaneous byte lanes without elision.
    const auto identities=[&](const SceneLightRecord& light) {
        if (!NativeSceneAnimationCompositionBudget::addLightIdentityBytes(light,8,bytes,error)) return false;
        if (!sceneAnimationBudget_->consumeLightIdentityWork(light,32))
            return reject(error,"camera/light composition named light identity copy work exceeds capacity");
        return true;
    };
    if (request.runtimeInputs.lightOverride && !identities(*request.runtimeInputs.lightOverride)) return false;
    for (std::size_t i=importedScene_->lightCount;i<callerLights.size();++i)
        if (!identities(callerLights[i])) return false;
    if (!add(request.runtimeInputs.lightOverrides.size(),sizeof(arbitgpu::NativeImportedLightOverride))) return false;
    if (!sceneAnimationBudget_->work(importedScene_->lightCount+scene_->lightCount,
            sizeof(SceneLightRecord),error)) return false;
    for (std::size_t i=0;i<importedScene_->lightCount;++i)
        if (!identities(importedScene_->lights[i])) return false;
    for (std::size_t i=importedScene_->lightCount;i<scene_->lightCount;++i)
        if (!identities(scene_->lights[i])) return false;
    const auto charge=NativeSceneAnimationCompositionBudget::reserve(sceneAnimationBudget_,bytes,error);
    if (!charge) return false;

    std::vector<std::uint8_t> required(nodes,0),objects(nodes,0);
    const auto ancestry=[&](std::uint64_t id,std::vector<std::uint8_t>& table) {
        if (!id || id>nodes) return false;
        for (std::optional<std::size_t> node=static_cast<std::size_t>(id-1);node;node=baseScene_->nodes[*node].parent)
        {
            if (*node>=nodes || table[*node]==2) return false;
            table[*node]=2;
        }
        for (auto& entry:table) if (entry==2) entry=1;
        return true;
    };
    for (std::size_t i=0;i<importedScene_->cameraCount;++i)
        if (!ancestry(importedScene_->cameras[i].id.value,required))
            return reject(error,"camera/light composition camera ancestry is invalid");
    for (std::size_t i=0;i<importedScene_->lightCount;++i)
        if (!ancestry(importedScene_->lights[i].id.value,required))
            return reject(error,"camera/light composition light ancestry is invalid");
    for (std::size_t i=0;i<importedScene_->objectCount;++i)
    {
        const auto id=importedScene_->objects[i].id.value;
        if (!id || !ancestry((id-1u)/65536u+1u,objects))
            return reject(error,"camera/light composition object ancestry is invalid");
    }
    for (const auto& track:evaluation.sceneAnimation->tracks())
    {
        const auto id=track.target().value;
        if (!id || id>nodes || !required[id-1] || objects[id-1]
            || track.channel()==visualanimation::Channel::MorphWeights
            || track.values().size()!=(track.channel()==visualanimation::Channel::Rotation?4u:3u))
            return reject(error,"camera/light composition requires camera/light-only tracks without rigid object deformation");
    }
    const auto& sample=*evaluation.sceneAnimation;
    const auto selected=std::find_if(document->clips.begin(),document->clips.end(),
        [&](const auto& value){return value.clip && value.clip->id()==sample.clip();});
    if (selected==document->clips.end() || !selected->jointBindings.empty() || !selected->morphBindings.empty())
        return reject(error,"camera/light composition selected clip requires deformation");

    auto runtime=request.runtimeInputs;
    if (request.sourceCameraNodeId && runtime.cameraOverride)
    {
        const auto* authored = visual3d_detail::findById(proceduralScene_->cameras,
            proceduralScene_->cameraCount, proceduralScene_->activeCamera);
        if (!sceneAnimationBudget_->work(proceduralScene_->cameraCount+scene_->cameraCount,
                sizeof(SceneCameraRecord),error)) return false;
        if (!authored || !scenecomposition::sameCameraRecord(*authored, *runtime.cameraOverride))
            return reject(error, "composed camera override does not match its authored active record");
        auto mapped = *authored;
        mapped.id.value |= 0x80000000u;
        const auto* retained = visual3d_detail::findById(scene_->cameras, scene_->cameraCount, mapped.id);
        if (!retained || !scenecomposition::sameCameraRecord(mapped, *retained))
            return reject(error, "composed camera override has no exact private mapped record");
        runtime.cameraOverride = *retained;
    }

    runtime.previousMotion.reset();runtime.compositionFrame.reset();
    if (!runtime.noteInstanceMapping) runtime.canonicalBlockCFrame.reset();
    std::vector<SceneCameraRecord> cameras;
    if (!gltf::sampleGlbCameraLights(*baseScene_,sample,operation.combinationMode,
            static_cast<float>(evaluation.playback.weight),*importedScene_,cameras,runtime.animatedLights,error)
        || !importedscene::evaluateImportedLights(request.importedLights,
            importedscene::importedLightDefinitions(*baseScene_,*importedScene_),*importedScene_,
            runtime.animatedLights,error)) return false;
    const auto camera=std::find_if(cameras.begin(),cameras.end(),
        [&](const auto& value){return request.sourceCameraNodeId ? value.id.value==*request.sourceCameraNodeId : value.id==scene_->activeCamera;});
    if (camera!=cameras.end()) runtime.animatedCamera=*camera;
    if (request.sourceCameraNodeId && camera == cameras.end())
        return reject(error, "selected imported camera has no exact sampled record");
    // Without a leaf selection, the procedural snapshot owns its active camera.
    // Imported identities stay intact. Procedural camera/light identities use
    // the same high-half namespace as procedural objects in this distinct lane.
    if (runtime.animatedLights.size()!=importedScene_->lightCount)
        return reject(error,"camera/light composition sampled light count differs from the imported light count");
    if (callerLights.size()>importedScene_->lightCount)
        runtime.animatedLights.insert(runtime.animatedLights.end(),
            callerLights.begin()+static_cast<std::ptrdiff_t>(importedScene_->lightCount),callerLights.end());
    else
        for (std::size_t i=importedScene_->lightCount;i<scene_->lightCount;++i)
            runtime.animatedLights.push_back(scene_->lights[i]);
    // Overrides stay unapplied here. The backend reader applies them once, to the
    // record each light index reads, and identity compares the same unapplied list.
    const auto seconds=static_cast<double>(request.frame.frame)*request.frame.rateDenominator/request.frame.rateNumerator;
    if (!std::isfinite(seconds) || std::abs(seconds)>surfacematerial::kMaximumEvaluationMagnitude)
        return reject(error,"camera/light composition rational project time is out of bounds");
    runtime.timeSeconds=static_cast<float>(seconds);
    runtime.objectNodeStableId=operation.pose.nodeStableId;
    runtime.morphWeight=operation.pose.morphEnabled || operation.combinationMode==visualanimation::CombinationMode::WeightedBlend
        ?1.0f:static_cast<float>(operation.playback.weight);

    auto sourceOwner=sceneAnimationOwner_;
    if (!sourceOwner || sourceOwner->scene()!=scene_ || sourceOwner->payload()!=admittedIdentity_.payload
        || sourceOwner->structuralRevision()!=request.structuralRevision
        || sourceOwner->evaluationRevision()!=request.evaluationRevision
        || sourceOwner->projectGeneration()!=request.projectGeneration
        || sourceOwner->helperGeneration()!=request.helperGeneration || sourceOwner->renderClipId()!=request.renderClipId
        || sourceOwner->sourceCameraNodeId()!=request.sourceCameraNodeId
        || sourceOwner->selection().sourceStableId!=operation.sourceStableId
        || sourceOwner->selection().deformationStableId!=operation.deformationStableId)
    {
        auto source=std::make_shared<arbitgpu::NativeDeformationScene>();
        source->sourceStableId=operation.sourceStableId;source->deformationStableId=operation.deformationStableId;
        source->structuralRevision=request.structuralRevision;source->evaluationRevision=request.evaluationRevision;
        source->clip=sample.clip();source->scene=scene_;
        auto candidate=std::shared_ptr<NativeSceneAnimationCompositionOwner>(new NativeSceneAnimationCompositionOwner);
        candidate->charge_=charge;
        candidate->payload_=retainSceneAnimationAllocation(admittedIdentity_.payload,sceneAnimationAdmissionCharge_);
        candidate->document_=baseScene_;
        candidate->animation_=retainSceneAnimationAllocation(document,sceneAnimationAdmissionCharge_);
        candidate->imported_=importedScene_;
        candidate->procedural_=retainSceneAnimationAllocation(proceduralScene_,sceneAnimationAdmissionCharge_);
        candidate->scene_=scene_;candidate->selection_=operation;
        candidate->sourceCameraNodeId_=request.sourceCameraNodeId;
        candidate->structural_=request.structuralRevision;candidate->evaluation_=request.evaluationRevision;
        candidate->project_=request.projectGeneration;candidate->helper_=request.helperGeneration;
        candidate->renderClip_=request.renderClipId;
        using Mapping=NativeSceneAnimationCompositionOwner::RecordMapping;
        const auto mappings=[&](const auto& records,std::size_t count,Mapping::Kind kind,bool imported) {
            for (std::size_t i=0;i<count;++i)
            {
                const auto id=records[i].id.value;
                candidate->mappings_[candidate->mappingCount_++]={kind,id,imported?id:id|0x80000000u,imported};
            }
        };
        mappings(importedScene_->cameras,importedScene_->cameraCount,Mapping::Kind::Camera,true);
        mappings(importedScene_->lights,importedScene_->lightCount,Mapping::Kind::Light,true);
        mappings(proceduralScene_->cameras,proceduralScene_->cameraCount,Mapping::Kind::Camera,false);
        mappings(proceduralScene_->lights,proceduralScene_->lightCount,Mapping::Kind::Light,false);
        candidate->source_=retainSceneAnimationAllocation<arbitgpu::NativeDeformationScene>(std::move(source),charge);
        sourceOwner=std::move(candidate);
    }
    const auto material=materialScene_.lock()==scene_?lastGoodMaterial_:nullptr;
    surfacematerial::MaterialEvaluationInputs materialInputs;materialInputs.timeSeconds=runtime.timeSeconds;
    videorender::fixture3d::RenderedFrame rendered;
    loopstep::Scope renderScope (use == ImportedAnimationEvaluationOwner::Preview
        ? "advance:imported-scene:animated-render" : "advance:imported-scene:animated-export-render");
    const bool success=use==ImportedAnimationEvaluationOwner::Preview
        ?sceneRenderer_.renderPreview(scene_,material,materialInputs,runtime,{request.width,request.height},
            videorender::fixture3d::kNativeGpuCapability,rendered,error)
        :sceneRenderer_.renderExport(scene_,material,materialInputs,runtime,{request.width,request.height},
            videorender::fixture3d::kNativeGpuCapability,rendered,error);
    if (!success) return false;
    if (rendered.pending)
    {
        // Retry state. The owner, frame history and caller receipt stay as they were.
        // Export never sees pending, so it refuses instead.
        if (use != ImportedAnimationEvaluationOwner::Preview)
            return reject(error,"camera/light composition export render returned pending");
        output.pending=true;
        error.clear();
        return true;
    }
    auto issued=std::shared_ptr<NativeSceneAnimationCompositionFrame>(new NativeSceneAnimationCompositionFrame);
    issued->charge_=charge;issued->owner_=sourceOwner;issued->sample_=evaluation.sceneAnimation;
    issued->nativeFrame_=rendered.nativeFrame;issued->material_=material?material->nativeProgram():nullptr;
    issued->operation_=operation;issued->sourceCameraNodeId_=request.sourceCameraNodeId;issued->playback_=evaluation.playback;issued->time_=request.frame;
    issued->use_=use;issued->width_=request.width;issued->height_=request.height;
    issued->camera_=runtime.animatedCamera;issued->lights_=runtime.animatedLights;
    issued->lightOverrides_=runtime.lightOverrides;
    issued->cameras_=std::move(cameras);
    issued->cameraOverride_=runtime.cameraOverride;issued->lightOverride_=runtime.lightOverride;issued->controls_=request.importedLights;
    issued->timeSeconds_=runtime.timeSeconds;issued->imageOutput_=runtime.imageOutput;
    issued->rawExportMask_=runtime.rawExportMask;issued->linearColor_=runtime.linearColor;
    ImportedAnimatedSceneReceipt receipt;
    receipt.sourceCameraNodeId=request.sourceCameraNodeId;
    receipt.owner=use;receipt.payload=sourceOwner->payload();receipt.source=sourceOwner->source();
    receipt.evaluation=std::move(evaluation);receipt.runtimeInputs=std::move(runtime);
    receipt.frame.use=use==ImportedAnimationEvaluationOwner::Preview
        ?videorender::animation3d::RenderUse::Preview:videorender::animation3d::RenderUse::Export;
    receipt.frame.source=receipt.source;receipt.frame.nativeFrame=std::move(rendered.nativeFrame);
    receipt.frame.stats.drawCount=rendered.stats.drawCount;receipt.frame.stats.time=request.frame;
    receipt.frame.stats.sourceStableId=operation.sourceStableId;receipt.frame.stats.deformationStableId=operation.deformationStableId;
    receipt.frame.stats.revision=request.structuralRevision;receipt.frame.stats.clipId=sample.clip().value;
    receipt.frame.stats.staticUploadCount=rendered.stats.staticUploadCount;
    receipt.frame.stats.reusedStaticResources=rendered.stats.reusedStaticResources;
    receipt.sceneAnimationComposition=issued;receipt.usedLastGoodMaterial=usedLastGoodMaterial;
    receipt.materialDiagnostic=std::move(materialDiagnostic);
    if (!receipt.valid()) return reject(error,"camera/light composition native frame does not match its issued authority");
    sceneAnimationOwner_=std::move(sourceOwner);source_=receipt.source;
    sceneAnimationFrames_[use==ImportedAnimationEvaluationOwner::Preview?0:1]=std::move(issued);
    output=std::move(receipt);error.clear();return true;
}

bool ImportedAnimatedScenePayloadExecution::chargeParentInspection() const noexcept
{
    return sceneAnimationBudget_ && sceneAnimationBudget_->consumeWork(1,16);
}

bool ImportedAnimatedScenePayloadExecution::inspectParentReceipt(const ImportedAnimatedSceneReceipt& receipt,
    const ImportedAnimatedSceneRequest& expected,ImportedAnimationEvaluationOwner use) const noexcept
{
    // Includes missing/private-association refusal. The surviving executor ledger
    // never resets after an inspector failure or a same-frame retry.
    if (!chargeParentInspection()) return false;
    return expected.parentIntent && receipt.inspectParent(*expected.parentIntent,expected.frame,expected.structuralRevision,
        expected.evaluationRevision,expected.projectGeneration,expected.helperGeneration,expected.renderClipId,use,
        expected.width,expected.height,&expected.operation,expected.proceduralScene,expected.sourceCameraNodeId,expected.runtimeInputs.cameraOverride,&expected.runtimeInputs);
}

void ImportedAnimatedScenePayloadExecution::clearSingleLeafState() noexcept
{
    clearSurfaceMaterial();
    source_.reset();
    parentOwner_.reset();
    sceneAnimationFrames_ = {}; sceneAnimationOwner_.reset();
    scene_.reset();
    importedScene_.reset();
    proceduralScene_.reset();
    baseScene_.reset();
    // Release the copy's storage, not just its elements, before its reservation.
    std::vector<gltf::GlbDeformationRenderBinding>().swap(bindings_);
    bindingCopyCharge_.reset();
    if (sceneAnimationAdmissionCharge_) consumer_ = {};
    admittedIdentity_ = {};
    sceneAnimationAdmissionCharge_.reset();
    motion_ = {}; motionOwner_.reset(); motionDevice_.clear();
}

void ImportedAnimatedScenePayloadExecution::reset() noexcept
{
    clearSingleLeafState();
    compositionLeaves_.clear();
    compositionEpochPayload_.reset(); compositionEpochIdentity_={}; highestCompositionEvaluationRevision_=0;
}

bool ImportedAnimatedScenePayloadExecution::admit(
    const visualmodelassetpayload::PayloadPtr& payload,
    const visualanimationimport::Request& operation,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& proceduralScene,
    std::optional<std::uint32_t> sourceCameraNodeId,
    const ImportedAnimatedSceneRequest& request,
    std::string& error)
{
    // The execute preflight reserved all live intent/identity/rollback copies.
    if ((request.parentIntent || admittedIdentity_.parentIntent)
        && !sceneAnimationBudget_->work(2*sizeof(visualimportedscenerender::ImportedParentIntentV1),2,error)) return false;
    const ImportedAnimatedSceneAdmissionIdentity identity {
        payload, operation.sceneIndex, operation.meshStableId,
        operation.animationClipStableId, operation.crossfadeClipStableId,
        operation.clipName, operation.crossfadeClipName, sourceCameraNodeId,
        request.parentIntent ? std::optional<visualimportedscenerender::ImportedParentIntentV1>{*request.parentIntent} : std::nullopt
    };
    if (identity == admittedIdentity_ && proceduralScene == proceduralScene_
        && (!request.parentIntent || (parentOwner_ && parentOwner_->structural_==request.structuralRevision
            && parentOwner_->evaluation_==request.evaluationRevision && parentOwner_->project_==request.projectGeneration
            && parentOwner_->helper_==request.helperGeneration && parentOwner_->renderClip_==request.renderClipId
            && parentOwner_->operation_.sourceStableId==operation.sourceStableId
            && parentOwner_->operation_.deformationStableId==operation.deformationStableId)))
    {
        if (request.parentIntent && !parentOwner_->inspect(*request.parentIntent))
            return reject(error,"imported parent current source changed or its inspection allowance is exhausted");
        return true;
    }
    // An epoch/intent change must not re-admit a mutated alias as fresh source.
    // A new source pointer or exact payload is a separate authored admission.
    if (request.parentIntent && parentOwner_ && proceduralScene==proceduralScene_
        && payload==admittedIdentity_.payload
        && !parentOwner_->inspect(parentOwner_->intent_))
        return reject(error,"imported parent original procedural source changed or its inspection allowance is exhausted");
    if (request.parentIntent && (!proceduralScene || request.parentIntent->composeStableId!=proceduralScene->id.value
        || request.parentIntent->importedSourceStableId!=operation.sourceStableId))
        return reject(error,"imported parent intent names a foreign compose or imported leaf");
    ImportedAnimationDeformationConsumer candidateConsumer;
    if (!candidateConsumer.admit(
            operation, payload->bytes().data(), payload->bytes().size(), error))
        return false;

    auto options = nativeImportedAnimationDecodeOptions(
        payload->bytes().size(), operation.sceneIndex);
    auto decodedScene = gltf::decodeAnimatedGlbBaseScene(
        payload->bytes(), options.admission, error);
    if (!decodedScene)
    {
        error = "imported animated scene base geometry rejected: " + error;
        return false;
    }
    const auto selectedMeshIndex = operation.meshStableId == 0
        ? std::optional<std::size_t> {}
        : std::optional<std::size_t> {
            static_cast<std::size_t>(operation.meshStableId - 1u) };
    auto adapted = gltf::adaptAnimatedGlbMeshToVisual3DSceneHeap(
        *decodedScene, error, selectedMeshIndex, true);
    if (!adapted)
    {
        error = "imported animated scene native adaptation rejected: " + error;
        return false;
    }
    if (proceduralScene && sourceCameraNodeId && !sceneAnimationBudget_->work(
            HarmonicMIDI::grid::Visual3DScene::kMaxCameras, sizeof(HarmonicMIDI::grid::SceneCameraRecord), error)) return false;
    if (sourceCameraNodeId && !sourceCameraRecord(*decodedScene, *adapted, *sourceCameraNodeId))
        return reject(error, "selected imported camera is not a genuine camera in the admitted source scene");
    auto document = candidateConsumer.admittedDocument();
    if (!document)
        return reject(error, "imported animated scene clip data is unavailable");
    std::vector<gltf::GlbDeformationRenderBinding> bindings;
    if (!collectImportedLeafBindings(*document, *adapted, operation.meshStableId, bindings, error)) return false;
    if (request.parentIntent && (bindings.empty() || !document->deformation))
        return reject(error,"imported parenting requires genuine source-bound skin or morph deformation");
    if (bindings.empty() && std::none_of(document->clips.begin(), document->clips.end(),
            [](const auto& clip) { return !clip.sceneTransformTargets.empty(); }))
        return reject(error, "imported animated scene has no selected mesh, camera or light animation bindings");
    if (!importedLeafDeformedNodesAreBound(*decodedScene, *document, *adapted, bindings, error)) return false;
    // The kept bindings copy their joint and morph vectors out of the
    // animation document, as a composition leaf's do, so the copy reserves
    // its bytes in the same ledger before it is kept. Only the byte ledger is
    // charged here: the plain path draws no lifetime work, so repeated
    // re-admissions of one leaf cannot exhaust it. A refusal destroys the
    // copy and its reservation together; the admitted state is unchanged.
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> bindingCopy;
    {
        using Budget = NativeSceneAnimationCompositionBudget;
        std::size_t copied = 0;
        const auto addCopy = [&](std::size_t count, std::size_t stride) {
            if (count > (Budget::kMaximumBytes - copied) / stride)
                return reject(error, "imported animated scene render binding copy exceeds logical CPU capacity");
            copied += count * stride; return true;
        };
        if (!addCopy(bindings.capacity(), sizeof(gltf::GlbDeformationRenderBinding))) return false;
        for (const auto& binding : bindings)
            if (!addCopy(binding.jointBaseTransforms.capacity(), sizeof(gltf::GlbJointBaseTransform))
                || !addCopy(binding.morphBaseWeights.capacity(), sizeof(float))) return false;
        if (copied != 0)
        {
            if (!sceneAnimationBudget_) sceneAnimationBudget_ = std::make_shared<Budget>();
            bindingCopy = Budget::reserve(sceneAnimationBudget_, copied, error);
            if (!bindingCopy)
                return reject(error, "imported animated scene render binding copy exceeds logical CPU capacity");
        }
    }

    auto imported = std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>(std::move(adapted));
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> animationAdmission;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed;
    if (proceduralScene && bindings.empty())
    {
        using namespace HarmonicMIDI::grid;
        // Released decode/adaptation owns its own admission. This census admits
        // every newly retained input and the composed-scene/caller copies.
        std::size_t bytes = 0, work = 0;
        if (!compositionInputBytes(payload,*decodedScene,*document,*imported,*proceduralScene,
                *proceduralScene,bytes,work,NativeSceneAnimationCompositionBudget::kMaximumWork,error,
                sceneAnimationBudget_)) return false;
        const auto extra = 2*sizeof(Visual3DScene)+sizeof(NativeSceneAnimationCompositionOwner)
            +sizeof(arbitgpu::NativeDeformationScene)+64*sizeof(std::shared_ptr<const void>);
        const auto add=[&](std::size_t count,std::size_t stride) {
            if (bytes>NativeSceneAnimationCompositionBudget::kMaximumBytes
                || count>(NativeSceneAnimationCompositionBudget::kMaximumBytes-bytes)/stride)
                return reject(error,"camera/light composition input exceeds logical CPU capacity");
            bytes+=count*stride;return true;
        };
        if (!add(1,extra) || !add(imported->textureTexels.capacity(),3*sizeof(SceneTexelRgba8))
            || !add(proceduralScene->textureTexels.capacity(),2*sizeof(SceneTexelRgba8))
            || !add(operation.asset.id.size()+operation.asset.contentSha256.size()+operation.asset.sourceMediaType.size()
                +operation.clipName.size()+operation.crossfadeClipName.size()+5,2)
            || !sceneAnimationBudget_->work(bytes,4,error)) return false;
        // In addition to retained originals, assembly copies all slots into
        // mapped, geometry, leaf and full scenes, then replaces the full light
        // table and appends active procedural records. Admit both identity
        // operands for appendScene comparisons before entering that routine.
        for (const auto& light:imported->lights)
            if (!NativeSceneAnimationCompositionBudget::addLightIdentityBytes(light,4,bytes,error)
                || !sceneAnimationBudget_->consumeLightIdentityWork(light,4*(Visual3DScene::kMaxLights+8))) return false;
        for (const auto& light:proceduralScene->lights)
            if (!NativeSceneAnimationCompositionBudget::addLightIdentityBytes(light,5,bytes,error)
                || !sceneAnimationBudget_->consumeLightIdentityWork(light,4*(Visual3DScene::kMaxLights+8))) return false;
        animationAdmission=NativeSceneAnimationCompositionBudget::reserve(sceneAnimationBudget_,bytes,error);
        if (!animationAdmission) return false;
        if (imported->cameraCount>Visual3DScene::kMaxCameras || imported->lightCount>Visual3DScene::kMaxLights
            || proceduralScene->cameraCount>Visual3DScene::kMaxCameras-imported->cameraCount
            || proceduralScene->lightCount>Visual3DScene::kMaxLights-imported->lightCount)
            return reject(error,"camera/light composition exceeds the full scene record capacity");
        auto mapped=std::make_shared<Visual3DScene>(*proceduralScene);
        for (std::size_t i=0;i<mapped->cameraCount;++i)
        {
            auto& id=mapped->cameras[i].id;
            if (!id.isValid() || id.value>=0x80000000u)
                return reject(error,"camera/light composition procedural camera identity exceeds its namespace");
            id.value|=0x80000000u;
        }
        if (mapped->activeCamera.isValid()) mapped->activeCamera.value|=0x80000000u;
        for (std::size_t i=0;i<mapped->lightCount;++i)
        {
            auto& id=mapped->lights[i].id;
            if (!id.isValid() || id.value>=0x80000000u)
                return reject(error,"camera/light composition procedural light identity exceeds its namespace");
            id.value|=0x80000000u;
        }
        auto geometry=gltf::composeImportedVisual3DScene(*mapped,*imported,error);
        if (!geometry) return false;
        auto full=std::make_shared<Visual3DScene>(*geometry);
        full->cameras=imported->cameras;full->cameraCount=imported->cameraCount;
        full->activeCamera=mapped->activeCamera;
        full->lights=imported->lights;full->lightCount=imported->lightCount;
        for (std::size_t i=0;i<mapped->cameraCount;++i) full->cameras[full->cameraCount++]=mapped->cameras[i];
        for (std::size_t i=0;i<mapped->lightCount;++i) full->lights[full->lightCount++]=mapped->lights[i];
        if (!validateVisual3DScene(*full).valid())
            return reject(error,"camera/light composition full scene is invalid");
        composed=retainSceneAnimationAllocation<Visual3DScene>(std::move(full),animationAdmission);
        imported=retainSceneAnimationAllocation(std::move(imported),animationAdmission);
    }
    else
    {
        std::shared_ptr<const NativeCompositionRetentionBudget::Charge> parentAssembly;
        if (request.parentIntent)
        {
            if (proceduralScene->objectCount>64 || imported->objectCount>64
                || proceduralScene->objectCount>64-imported->objectCount)
                return reject(error,"imported parent aggregate object count exceeds capacity");
            std::size_t p=0,i=0;
            if (!ImportedParentBindingOwner::sceneBytes(*proceduralScene,*sceneAnimationBudget_,p)
                || !ImportedParentBindingOwner::sceneBytes(*imported,*sceneAnimationBudget_,i))
                return reject(error,"imported parent assembly capacity exceeds logical CPU capacity");
            // Adapter candidate, imported leaf and simultaneously live returned
            // basis, including full inactive light identities and texel capacity.
            const auto bytes=2*p+2*i+sizeof(NativeCompositionRetentionBudget::Charge)
                +10*sizeof(std::shared_ptr<const void>);
            if (!sceneAnimationBudget_->work(bytes,2,error)) return false;
            parentAssembly=NativeCompositionRetentionBudget::reserve(compositionBudget_,bytes,error);
            if (!parentAssembly) return false;
        }
        composed = proceduralScene ? gltf::composeImportedVisual3DScene(*proceduralScene, *imported, error) : imported;
        if (composed && parentAssembly) composed=retainParentAllocation(composed,parentAssembly,sceneAnimationBudget_);
    }
    if (!composed) return false;
    std::shared_ptr<const ImportedParentBindingOwner> parentOwner;
    if (request.parentIntent)
    {
        using Scene=HarmonicMIDI::grid::Visual3DScene;
        using Owner=ImportedParentBindingOwner;
        std::size_t cloneBytes=0,importedBytes=0,proceduralBytes=0;
        if (!Owner::sceneBytes(*composed,*sceneAnimationBudget_,cloneBytes)
            || !Owner::sceneBytes(*imported,*sceneAnimationBudget_,importedBytes)
            || !Owner::sceneBytes(*proceduralScene,*sceneAnimationBudget_,proceduralBytes))
            return reject(error,"imported parent full scene capacity census exceeds logical CPU capacity");
        // Basis, parented clone, frozen procedural witness and both original
        // sources remain live through inspection. Thirteen alias calls count
        // their 3-field wrapper, parameter, temporary, local and return pointers
        // without reference-count or elision credit.
        std::size_t bytes=2*cloneBytes+importedBytes+2*proceduralBytes+sizeof(Owner)
            +2*sizeof(Owner::Parents)+64*sizeof(bool)+4*sizeof(std::size_t)
            +sizeof(NativeCompositionRetentionBudget::Charge)+13*10*sizeof(std::shared_ptr<const void>)
            +2*sizeof(std::shared_ptr<const Scene>)+sizeof(NativeCompositionRetentionBudget::Charge);
        const auto add=[&](std::size_t count,std::size_t stride) {
            if (bytes>NativeCompositionRetentionBudget::kMaximumBytes
                || count>(NativeCompositionRetentionBudget::kMaximumBytes-bytes)/stride)
                return reject(error,"imported parent retained source capacity exceeds logical CPU capacity");
            bytes+=count*stride;return true;
        };
        std::size_t originals=0,work=0;
        if (!compositionInputBytes(payload,*decodedScene,*document,*imported,*proceduralScene,*composed,
                originals,work,NativeSceneAnimationCompositionBudget::kMaximumWork,error,sceneAnimationBudget_)
            || !add(originals,1) || !add(operation.asset.id.capacity()+1,1) || !add(operation.asset.contentSha256.capacity()+1,1)
            || !add(operation.asset.sourceMediaType.capacity()+1,1)
            || !add(operation.clipName.capacity()+1,1) || !add(operation.crossfadeClipName.capacity()+1,1)
            || !sceneAnimationBudget_->work(bytes,1,error)) return false;
        auto charge=NativeCompositionRetentionBudget::reserve(compositionBudget_,bytes,error);
        if (!charge) return false;
        Owner::Parents parents;
        if (!Owner::parents(*request.parentIntent,*imported,*proceduralScene,*composed,*sceneAnimationBudget_,parents)
            || !sceneAnimationBudget_->work(cloneBytes+proceduralBytes,1,error)
            || !sceneAnimationBudget_->work(composed->objectCount,sizeof(HarmonicMIDI::grid::SceneObjectId),error))
            return reject(error,"imported parent complete source-root membership is invalid or exhausted");
        // Reservations precede their scene values so failure unwinding also
        // destroys oversized copies before refunding the additional debit.
        std::shared_ptr<const NativeCompositionRetentionBudget::Charge> overflow,witnessOverflow;
        auto witness=std::make_shared<Scene>(*proceduralScene);
        auto clone=std::make_shared<Scene>(*composed);
        std::size_t actual=0;
        if (!Owner::sceneBytes(*clone,*sceneAnimationBudget_,actual))
            return reject(error,"imported parent clone capacity census exhausted");
        // A library copy that requires more capacity cannot borrow credit.
        if (actual>cloneBytes)
        {
            overflow=NativeCompositionRetentionBudget::reserve(compositionBudget_,actual-cloneBytes,error);
            if (!overflow) return false;
        }
        std::size_t witnessActual=0;
        if (!Owner::sceneBytes(*witness,*sceneAnimationBudget_,witnessActual))
            return reject(error,"imported parent source witness capacity census exhausted");
        if (witnessActual>proceduralBytes)
        {
            witnessOverflow=NativeCompositionRetentionBudget::reserve(compositionBudget_,witnessActual-proceduralBytes,error);
            if (!witnessOverflow) return false;
        }
        if (!Owner::unchangedSource(*witness,*proceduralScene,*sceneAnimationBudget_))
            return reject(error,"imported parent frozen source witness changed original fields");
        for (std::size_t x=0;x<clone->objectCount;++x) clone->objects[x].parent=parents.values[x];
        if (!Owner::conservation(*composed,*clone,parents,*sceneAnimationBudget_))
            return reject(error,"imported parent clone changed fields outside the intended parent array");
        auto issued=std::shared_ptr<Owner>(new Owner);
        issued->retention_=charge;issued->budget_=sceneAnimationBudget_;
        issued->intent_=*request.parentIntent;issued->operation_=operation;
        if (!sceneAnimationBudget_->work(1,10,error) || !ImportedParentBindingFrame::capacitiesFit(issued->operation_,operation))
            return reject(error,"imported parent copied source identity exceeds its reserved capacities");
        issued->payload_=retainParentAllocation(payload,charge,sceneAnimationBudget_);
        issued->animation_=retainParentAllocation(document,charge,sceneAnimationBudget_);
        issued->imported_=retainParentAllocation(imported,charge,sceneAnimationBudget_);
        issued->procedural_=retainParentAllocation(proceduralScene,charge,sceneAnimationBudget_);
        std::shared_ptr<const Scene> sourceWitness=std::move(witness);
        if (witnessOverflow) sourceWitness=retainCompositionAllocation(std::move(sourceWitness),witnessOverflow);
        issued->proceduralWitness_=retainParentAllocation(std::move(sourceWitness),charge,sceneAnimationBudget_);
        issued->basis_=retainParentAllocation(composed,charge,sceneAnimationBudget_);
        std::shared_ptr<const Scene> frozen=std::move(clone);
        if (overflow) frozen=retainCompositionAllocation(std::move(frozen),overflow);
        issued->scene_=retainParentAllocation(std::move(frozen),charge,sceneAnimationBudget_);
        issued->structural_=request.structuralRevision;issued->evaluation_=request.evaluationRevision;
        issued->project_=request.projectGeneration;issued->helper_=request.helperGeneration;
        issued->renderClip_=request.renderClipId;
        composed=issued->scene_;imported=issued->imported_;parentOwner=std::move(issued);
    }
    clearSurfaceMaterial();
    // A single leaf replaces any admitted composition leaves and their charges.
    // A composed execute() restores them from its incumbent on refusal. Any
    // other execute() has no incumbent, by design: it renders no Scene
    // Compose, and a later multi-leaf frame admits its leaves again.
    compositionLeaves_.clear();
    consumer_ = std::move(candidateConsumer);
    importedScene_ = std::move(imported);
    proceduralScene_ = proceduralScene;
    scene_ = std::move(composed);
    baseScene_ = std::make_shared<const gltf::GlbStaticMeshDocument>(std::move(*decodedScene));
    if (animationAdmission) baseScene_=retainSceneAnimationAllocation(baseScene_,animationAdmission);
    if (parentOwner)
    {
        // Not escaped until its actual native owner has independently admitted.
        auto issuing=std::const_pointer_cast<ImportedParentBindingOwner>(parentOwner);
        const auto charge=std::static_pointer_cast<const NativeCompositionRetentionBudget::Charge>(issuing->retention_);
        baseScene_=retainParentAllocation(baseScene_,charge,sceneAnimationBudget_);
        issuing->document_=baseScene_;
    }
    parentOwner_=std::move(parentOwner);
    sceneAnimationAdmissionCharge_=std::move(animationAdmission);
    sceneAnimationOwner_.reset();sceneAnimationFrames_={};
    // Re-admission peaks at twice the binding copy: the candidate's
    // reservation was charged above while the incumbent's bindingCopyCharge_
    // is still live. The old copy is released before its reservation.
    bindings_ = std::move(bindings);
    bindingCopyCharge_ = std::move(bindingCopy);
    admittedIdentity_ = identity;
    source_.reset();
    motion_ = {}; motionOwner_.reset();
    error.clear();
    return true;
}

bool ImportedAnimatedScenePayloadExecution::admitCompositionLeaf(
    const std::shared_ptr<NativeSceneAnimationCompositionBudget>& budget,
    const CompositionLeafInput& input, CompositionLeaf& output, std::string& error)
{
    using HarmonicMIDI::grid::Visual3DScene;
    using Budget = NativeSceneAnimationCompositionBudget;
    const auto& payload = input.payload;
    const auto& operation = input.operation;
    if (!budget)
        return reject(error, "imported composition leaf has no scene animation ledger");
    if (!payload || payload->bytes().empty())
        return reject(error, "imported composition leaf exact payload is not resident");
    if (!input.animated && (operation.animationClipStableId != 0 || operation.crossfadeClipStableId != 0
            || !operation.clipName.empty() || !operation.crossfadeClipName.empty()))
        return reject(error, "imported composition static leaf names an animation clip");
    CompositionLeaf candidate;
    candidate.animated = input.animated;
    // Fixed storage is reserved before any decode: the leaf record, the
    // retained document and rest scene headers, the animation document header,
    // both reservations, alias wrappers and the leaf's copied request strings.
    std::size_t fixed = sizeof(CompositionLeaf) + sizeof(gltf::GlbStaticMeshDocument)
        + sizeof(gltf::GlbAnimationDocument) + sizeof(Visual3DScene)
        + candidate.charges.size() * sizeof(Budget::Charge) + 8 * sizeof(std::shared_ptr<const void>);
    const auto add = [&](std::size_t count, std::size_t stride) {
        if (fixed > Budget::kMaximumBytes || count > (Budget::kMaximumBytes - fixed) / stride)
            return reject(error, "imported composition leaf storage exceeds logical CPU capacity");
        fixed += count * stride; return true;
    };
    // Two copies of each string: the leaf request's and its consumer's.
    for (const auto* text : {&operation.asset.id, &operation.asset.contentSha256, &operation.asset.sourceMediaType,
            &operation.clipName, &operation.crossfadeClipName})
    {
        if (text->capacity() >= Budget::kMaximumBytes)
            return reject(error, "imported composition leaf storage exceeds logical CPU capacity");
        if (!add(text->capacity() + 1, 2)) return false;
    }
    if (!budget->work(fixed, 1, error)) return false;
    candidate.charges[0] = Budget::reserve(budget, fixed, error);
    if (!candidate.charges[0]) return false;

    std::shared_ptr<const gltf::GlbAnimationDocument> animation;
    if (input.animated)
    {
        if (!candidate.consumer.admit(operation, payload->bytes().data(), payload->bytes().size(), error))
            return false;
        animation = candidate.consumer.admittedDocument();
        if (!animation)
            return reject(error, "imported animated scene clip data is unavailable");
    }
    auto options = nativeImportedAnimationDecodeOptions(payload->bytes().size(), operation.sceneIndex);
    auto decodedScene = gltf::decodeAnimatedGlbBaseScene(payload->bytes(), options.admission, error);
    if (!decodedScene)
    {
        error = "imported composition leaf base geometry rejected: " + error;
        return false;
    }
    const auto selectedMeshIndex = operation.meshStableId == 0
        ? std::optional<std::size_t> {}
        : std::optional<std::size_t> {static_cast<std::size_t>(operation.meshStableId - 1u)};
    std::unique_ptr<Visual3DScene> adapted;
    if (input.animated)
        adapted = gltf::adaptAnimatedGlbMeshToVisual3DSceneHeap(*decodedScene, error, selectedMeshIndex, true);
    else if (auto still = gltf::adaptStaticGlbToVisual3DScene(*decodedScene, error, selectedMeshIndex))
        adapted = std::make_unique<Visual3DScene>(std::move(*still));
    if (!adapted)
    {
        error = "imported composition leaf native adaptation rejected: " + error;
        return false;
    }
    // The decoded closure is reserved before the leaf retains it. A leaf has
    // no procedural or composed scene, so empty scenes fill those operands.
    {
        static const gltf::GlbAnimationDocument noAnimation {};
        static const auto none = std::make_unique<const Visual3DScene>();
        std::size_t closure = 0, work = 0;
        if (!compositionInputBytes(payload, *decodedScene, animation ? *animation : noAnimation, *adapted,
                *none, *none, closure, work, Budget::kMaximumWork, error, budget)) return false;
        // The census starts from three scene headers and both document
        // headers. The fixed reservation already holds one rest scene and
        // both documents, and the leaf retains no other scene.
        // While incumbent leaves are alive, a re-admission counts a shared
        // payload here a second time. One leaf does the same: it reserves its
        // census before releasing sceneAnimationAdmissionCharge_. The double
        // count lasts only until the swap and fails closed.
        const auto counted = 3 * sizeof(Visual3DScene) + sizeof(gltf::GlbStaticMeshDocument)
            + sizeof(gltf::GlbAnimationDocument);
        if (closure < counted)
            return reject(error, "imported composition leaf census is inconsistent");
        closure -= counted;
        candidate.charges[1] = Budget::reserve(budget, closure, error);
        if (!candidate.charges[1]) return false;
    }
    if (input.animated)
    {
        if (!collectImportedLeafBindings(*animation, *adapted, operation.meshStableId, candidate.bindings, error))
            return false;
        if (candidate.bindings.empty() && std::none_of(animation->clips.begin(), animation->clips.end(),
                [](const auto& clip) { return !clip.sceneTransformTargets.empty(); }))
            return reject(error, "imported animated scene has no selected mesh, camera or light animation bindings");
        if (!importedLeafDeformedNodesAreBound(*decodedScene, *animation, *adapted, candidate.bindings, error))
            return false;
        // The kept bindings copy their joint and morph vectors out of the
        // animation document. The census above charged only the document's
        // own vectors, so the copy takes its own reservation before the leaf
        // keeps it. A refusal here destroys the copy with the candidate.
        std::size_t copied = 0;
        const auto addCopy = [&](std::size_t count, std::size_t stride) {
            if (count > (Budget::kMaximumBytes - copied) / stride)
                return reject(error, "imported composition leaf render binding copy exceeds logical CPU capacity");
            copied += count * stride; return true;
        };
        if (!addCopy(candidate.bindings.capacity(), sizeof(gltf::GlbDeformationRenderBinding))) return false;
        for (const auto& binding : candidate.bindings)
            if (!addCopy(binding.jointBaseTransforms.capacity(), sizeof(gltf::GlbJointBaseTransform))
                || !addCopy(binding.morphBaseWeights.capacity(), sizeof(float))) return false;
        if (copied != 0)
        {
            if (!budget->work(copied, 1, error)) return false;
            candidate.charges[2] = Budget::reserve(budget, copied, error);
            if (!candidate.charges[2])
                return reject(error, "imported composition leaf render binding copy exceeds logical CPU capacity");
        }
    }
    candidate.payload = payload;
    candidate.operation = operation;
    candidate.scene = std::shared_ptr<const Visual3DScene>(std::move(adapted));
    candidate.document = std::make_shared<const gltf::GlbStaticMeshDocument>(std::move(*decodedScene));
    output = std::move(candidate);
    error.clear();
    return true;
}

bool ImportedAnimatedScenePayloadExecution::admitCompositionLeaves(
    const std::vector<CompositionLeafInput>& inputs, std::string& error)
{
    using HarmonicMIDI::grid::Visual3DScene;
    using namespace HarmonicMIDI::grid::scenecomposition;
    if (inputs.size() < 2 || inputs.size() > kMaximumImportedLeaves)
        return reject(error, "imported composition requires two to eight leaves");
    if (std::none_of(inputs.begin(), inputs.end(), [](const auto& input) { return input.animated; }))
        return reject(error, "imported animated composition requires at least one animated leaf");
    if (!sceneAnimationBudget_) sceneAnimationBudget_ = std::make_shared<NativeSceneAnimationCompositionBudget>();
    // Candidates hold their own reservations; a refusal destroys them, and
    // with them every reservation, before returning. Admitted leaves keep
    // theirs until this succeeds, so a re-admission briefly needs room for
    // both sets in the 8 MiB ledger. Work drawn from the lifetime work ledger
    // is never refunded, for refused attempts too, as for one leaf.
    // The candidates vector is per-call staging, freed on return. Each leaf
    // record it holds is charged in that leaf's fixed reservation.
    std::vector<CompositionLeaf> candidates(inputs.size());
    std::size_t objects = 0, materials = 0, textures = 0, texels = 0;
    for (std::size_t leaf = 0; leaf < inputs.size(); ++leaf)
    {
        auto& candidate = candidates[leaf];
        if (!admitCompositionLeaf(sceneAnimationBudget_, inputs[leaf], candidate, error)) return false;
        const auto& document = *candidate.document;
        const auto& scene = *candidate.scene;
        // Each leaf keeps the single-leaf caps and its identities below the stride.
        if (!importedLeafCountFits(inputs.size(), document.nodes.size())
            || !importedLeafCountFits(inputs.size(), document.materials.size())
            || !importedLeafCountFits(inputs.size(), document.textures.size())
            || scene.objectCount > Visual3DScene::kMaxObjects || scene.materialCount > Visual3DScene::kMaxMaterials
            || scene.textureCount > Visual3DScene::kMaxTextures || scene.textureTexelCount > Visual3DScene::kMaxTextureTexels)
        {
            error = "imported composition leaf " + std::to_string(leaf)
                + " exceeds the per-leaf node, material or texture capacity";
            return false;
        }
        objects += scene.objectCount; materials += scene.materialCount;
        textures += scene.textureCount; texels += scene.textureTexelCount;
        // The composed scene holds every leaf's records at once. The sums
        // leave out the procedural scene; executeComposition's join through
        // appendScene and composeImportedVisual3DScene refuses any overflow
        // with it before a scene or owner is published.
        if (objects > Visual3DScene::kMaxObjects || materials > Visual3DScene::kMaxMaterials
            || textures > Visual3DScene::kMaxTextures || texels > Visual3DScene::kMaxTextureTexels)
            return reject(error, "imported composition leaves together exceed the scene object, material or texture capacity");
    }
    clearSingleLeafState();
    consumer_ = {};
    compositionLeaves_ = std::move(candidates);
    error.clear();
    return true;
}

bool ImportedAnimatedScenePayloadExecution::evaluateCompositionLeaf(ImportedAnimationEvaluationOwner owner,
    const CompositionLeaf& leaf, const visualanimationimport::Request& operation,
    const visualdeformation::RationalFrameTime& frame, std::uint64_t structuralRevision,
    ImportedAnimationDeformationEvaluation& evaluation, std::string& error)
{
    if (!leaf.payload || !leaf.animated || !leaf.consumer.admittedDocument())
        return reject(error, "imported composition leaf is not admitted");
    return owner == ImportedAnimationEvaluationOwner::Preview
        ? leaf.consumer.evaluatePreview(operation, frame, structuralRevision, evaluation, error)
        : leaf.consumer.evaluateExport(operation, frame, structuralRevision, evaluation, error);
}

bool ImportedAnimatedScenePayloadExecution::publishSurfaceMaterial(
    const surfacematerialbinding::ImportedSceneMaterialRequest& request,
    std::string& error)
{
    const auto rejectMaterial = [&] (std::string diagnostic)
    {
        if (request.programRevision != 0
            && request.programRevision == highestMaterialRevisionSeen_
            && request.programRevision > lastGoodMaterialRevision_)
            rejectedMaterialRevision_ = request.programRevision;
        error = "native animated material publication rejected: " + std::move(diagnostic);
        return false;
    };
    if (request.programRevision != 0
        && request.programRevision < highestMaterialRevisionSeen_)
        return rejectMaterial("surface material request revision is stale");
    if (request.programRevision > highestMaterialRevisionSeen_)
        highestMaterialRevisionSeen_ = request.programRevision;
    if (request.programRevision != 0
        && request.programRevision == rejectedMaterialRevision_)
        return rejectMaterial("surface material request revision was already rejected");
    if (scene_ == nullptr)
        return rejectMaterial("animated scene admission is empty");

    using videohelper::materialprogram::BackendTarget;
    const auto backendName = renderer_.backendInfo().backend;
    const auto target = backendName == "metal" ? BackendTarget::Metal
                      : backendName == "opengl" ? BackendTarget::OpenGl
                                                 : BackendTarget::Invalid;
    if (target == BackendTarget::Invalid)
        return rejectMaterial("native deformation backend has no material compiler target");

    std::string diagnostic;
    auto candidate = videorender::fixture3d::admitSurfaceMaterialBinding(
        scene_, request, target, diagnostic);
    if (candidate == nullptr)
        return rejectMaterial(std::move(diagnostic));

    const auto materialScene = materialScene_.lock();
    if (request.programRevision == lastGoodMaterialRevision_)
    {
        if (materialScene == scene_ && lastGoodMaterial_ != nullptr
            && lastGoodMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodMaterial_->structuralRevision() == candidate->structuralRevision()
            && lastGoodMaterial_->evaluationRevision() == candidate->evaluationRevision()
            && lastGoodMaterial_->bindingDigest() == candidate->bindingDigest())
        {
            error.clear();
            return true;
        }
        const bool nextEvaluation = materialScene == scene_ && lastGoodMaterial_ != nullptr
            && lastGoodMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodMaterial_->structuralRevision() == candidate->structuralRevision()
            && candidate->evaluationRevision() > lastGoodMaterial_->evaluationRevision();
        if (! nextEvaluation)
            return rejectMaterial(
                "surface material revision already names a different immutable request");
    }
    if (lastGoodMaterial_ != nullptr && materialScene == scene_
        && lastGoodMaterial_->sceneRevision() != candidate->sceneRevision())
        return rejectMaterial(
            "surface material scene revision does not match the immutable admission");

    materialScene_ = scene_;
    lastGoodMaterial_ = std::move(candidate);
    lastGoodMaterialRevision_ = request.programRevision;
    rejectedMaterialRevision_ = 0;
    error.clear();
    return true;
}

void ImportedAnimatedScenePayloadExecution::clearSurfaceMaterial() noexcept
{
    materialScene_.reset();
    lastGoodMaterial_.reset();
    highestMaterialRevisionSeen_ = 0;
    lastGoodMaterialRevision_ = 0;
    rejectedMaterialRevision_ = 0;
}

bool ImportedAnimatedScenePayloadExecution::execute(
    ImportedAnimationEvaluationOwner owner,
    const ImportedAnimatedSceneRequest& request,
    ImportedAnimatedSceneReceipt& output,
    std::string& error,
    ImportedSceneParentReceiptStaging* parentStaging)
{
    // Several imported leaves take their own route; one leaf never reaches it.
    if (!request.compositionLeaves.empty())
    {
        if (parentStaging || request.parentIntent)
            return reject(error, "Scene Compose refused: imported parenting with multiple imported leaves is not rendered yet");
        return executeComposition(owner, request, output, error);
    }
    // S5a admits one rigid animated leaf with a non-neutral object-local transform.
    // The uses below are refused by name. Leaf shape checks follow admission, once
    // the bindings and the drawn object are known.
    const bool localTransform = !visualimportedobjectlocal::same(request.objectLocalTransform, {});
    if (localTransform)
    {
        if (!visualimportedobjectlocal::valid(request.objectLocalTransform))
            return reject(error, "Imported object local transform is malformed");
        if (request.parentIntent)
            return reject(error, "Imported object local transforms with imported parenting are not rendered yet");
        if (request.proceduralScene)
            return reject(error, "Imported object local transforms with procedural composition are not rendered yet");
        if (request.sourceCameraNodeId)
            return reject(error, "Imported object local transforms with a selected source camera are not rendered yet");
        if (request.runtimeInputs.noteInstanceMapping)
            return reject(error, "Imported object local transforms with note instancing are not rendered yet");
        if (request.runtimeInputs.glassCaustic)
            return reject(error, "Imported object local transforms with glass caustics are not rendered yet");
        if (request.runtimeInputs.diffractionSourceProgram || request.runtimeInputs.diffractionEvaluatedProgram)
            return reject(error, "Imported object local transforms with diffraction gratings are not rendered yet");
        if (request.runtimeInputs.compositionFrame || request.runtimeInputs.objectWorlds)
            return reject(error, "Imported object local transforms cannot reuse a composition frame or caller object worlds");
    }
    const bool composition = request.proceduralScene != nullptr;
    if (request.sourceCameraNodeId && (!request.operation.sceneIndex || *request.sourceCameraNodeId == 0
        || *request.sourceCameraNodeId > 16384))
        return reject(error,"imported animated source camera selector is malformed");
    if (composition && (request.evaluationRevision==0 || request.evaluationRevision<request.structuralRevision))
        return reject(error,"native composition requires the actual nonzero parameter evaluation epoch");
    // The one-leaf caller receipt clears on every exit that neither publishes nor
    // stays pending. A pending attempt keeps the caller's last frame.
    output.pending = false;
    bool published=false;
    const bool clearOutputOnExit = !composition && !request.parentIntent;
    const auto clearUnpublishedOutput = [&] {
        if (clearOutputOnExit && !published && !output.pending) output = {};
    };
    struct ClearUnpublishedOutput final
    {
        const decltype(clearUnpublishedOutput)& action;
        ~ClearUnpublishedOutput() { action(); }
    } clearUnpublishedOutputOnExit {clearUnpublishedOutput};
    // Preserve the existing legacy ordering. Composition adds a bounded outer
    // transaction; backend caches remain guarded by their exact source keys.
    if (composition && !compositionBudget_) compositionBudget_=std::make_shared<NativeCompositionRetentionBudget>();
    if ((composition || request.sourceCameraNodeId) && !sceneAnimationBudget_) sceneAnimationBudget_=std::make_shared<NativeSceneAnimationCompositionBudget>();
    std::shared_ptr<const NativeCompositionRetentionBudget::Charge> parentStagingCharge;
    if (request.parentIntent || (parentOwner_ && composition))
    {
        if (!composition) return reject(error,"imported parent intent requires its procedural composition");
        // Requests borrow the fixed input. These are all newly live value copies:
        // admission identity, its initializer/optional, rollback and restored
        // identity, owner capture, expected-parent scratch, receipt/frame staging.
        constexpr auto bytes=7*sizeof(visualimportedscenerender::ImportedParentIntentV1)
            +3*sizeof(ImportedParentBindingOwner::Parents)+64*sizeof(bool)+4*sizeof(std::size_t)
            +sizeof(ImportedAnimatedSceneReceipt)+kImportedAnimatedSceneRequestChargeBytes
            +sizeof(NativeCompositionRetentionBudget::Charge)+16*sizeof(std::shared_ptr<const void>);
        if (!sceneAnimationBudget_->work(bytes,2,error)) return false;
        parentStagingCharge=NativeCompositionRetentionBudget::reserve(compositionBudget_,bytes,error);
        if (!parentStagingCharge) return false;
        if (request.parentIntent && (request.runtimeInputs.noteInstanceMapping || request.runtimeInputs.glassCaustic
            || request.runtimeInputs.compositionFrame || request.runtimeInputs.previousMotion
            || request.runtimeInputs.animatedCamera
            || (request.material && request.material->vertexModifier)))
            return reject(error,"imported parenting refuses notes, foreign frames and material vertex processors");
    }
    std::size_t callerLogicalWork=0;
    constexpr auto callerWorkLimit=8*arbitgpu::NativeDeformationCompositionOwner::kMaximumLogicalWork;
    const auto chargeCallerWork=[&](std::size_t count,std::size_t steps) {
        if (count>(callerWorkLimit-callerLogicalWork)/steps)
            return reject(error,"native composition named caller work exceeds capacity");
        callerLogicalWork+=count*steps;return true;
    };
    std::shared_ptr<const NativeCompositionRetentionBudget::Charge> rollbackCharge;
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> animationRollbackCharge;
    if (composition)
    {
        // The leaf list header is outside the released single-leaf copy; it is
        // charged below only when several leaves are admitted.
        std::size_t rollbackBytes=sizeof(std::optional<ImportedAnimatedScenePayloadExecution>)
            -sizeof(std::vector<CompositionLeaf>)+sizeof(NativeCompositionRetentionBudget::Charge)
            +8*sizeof(std::shared_ptr<const void>)+2*4096*sizeof(bool);
        const auto charge=[&](std::size_t count,std::size_t stride) {
            if (count>(NativeCompositionRetentionBudget::kMaximumBytes-rollbackBytes)/stride)
                return reject(error,"native composition rollback staging exceeds logical CPU capacity");
            rollbackBytes+=count*stride;return true;
        };
        if (!charge(admittedIdentity_.clipName.size()+1,1) || !charge(admittedIdentity_.crossfadeClipName.size()+1,1)
            || !charge(motionDevice_.size()+1,1) || !charge(motion_[0].mapping.size()+1,1)
            || !charge(motion_[1].mapping.size()+1,1) || !charge(bindings_.size(),sizeof(gltf::GlbDeformationRenderBinding))) return false;
        if (admittedIdentity_.payload)
        {
            const auto& key=admittedIdentity_.payload->key();
            if (!charge(key.id.size()+1,1) || !charge(key.contentSha256.size()+1,1) || !charge(key.sourceMediaType.size()+1,1)) return false;
        }
        for (const auto& binding:bindings_)
            if (!charge(binding.jointBaseTransforms.size(),sizeof(gltf::GlbJointBaseTransform))
                || !charge(binding.morphBaseWeights.size(),sizeof(float))) return false;
        // The incumbent copies every admitted leaf tuple: its request strings,
        // its consumer's exact asset key and its render binding vectors.
        // An empty list (one leaf) adds nothing, not even its header.
        if (!compositionLeaves_.empty() && !charge(1,sizeof(std::vector<CompositionLeaf>))) return false;
        if (!charge(compositionLeaves_.size(),sizeof(CompositionLeaf))) return false;
        for (const auto& leaf:compositionLeaves_)
        {
            const auto& operation=leaf.operation;
            if (!charge(operation.clipName.size()+1,1) || !charge(operation.crossfadeClipName.size()+1,1)
                || !charge(leaf.bindings.size(),sizeof(gltf::GlbDeformationRenderBinding))) return false;
            // Two key copies: the leaf request's and its consumer's.
            for (int copy=0;copy<2;++copy)
                if (!charge(operation.asset.id.size()+1,1) || !charge(operation.asset.contentSha256.size()+1,1)
                    || !charge(operation.asset.sourceMediaType.size()+1,1)) return false;
            for (const auto& binding:leaf.bindings)
                if (!charge(binding.jointBaseTransforms.size(),sizeof(gltf::GlbJointBaseTransform))
                    || !charge(binding.morphBaseWeights.size(),sizeof(float))) return false;
        }
        // Byte-lane copies bound caller staging and worst-case restoration;
        // object headers and vector element copies are both charged. Library
        // internals and control blocks remain outside this logical schedule.
        if (!chargeCallerWork(rollbackBytes,2)) return false;
        rollbackCharge=NativeCompositionRetentionBudget::reserve(compositionBudget_,rollbackBytes,error);
        if (!rollbackCharge) return false;
        if (sceneAnimationAdmissionCharge_)
        {
            if (!sceneAnimationBudget_->work(rollbackBytes,2,error)) return false;
            animationRollbackCharge=NativeSceneAnimationCompositionBudget::reserve(sceneAnimationBudget_,rollbackBytes,error);
            if (!animationRollbackCharge) return false;
        }
    }
    // Copy only on the newly enabled path. References to immutable authorities
    // and native resources stay shared; the preceding reservation covers the
    // copied control strings and binding vectors.
    std::optional<ImportedAnimatedScenePayloadExecution> incumbent;
    if (composition) incumbent.emplace(*this);
    const auto restore=[&] {
        if (!incumbent || published) return;
        consumer_=std::move(incumbent->consumer_);
        admittedIdentity_=std::move(incumbent->admittedIdentity_);
        scene_=std::move(incumbent->scene_); importedScene_=std::move(incumbent->importedScene_);
        proceduralScene_=std::move(incumbent->proceduralScene_); baseScene_=std::move(incumbent->baseScene_);
        bindings_=std::move(incumbent->bindings_); bindingCopyCharge_=std::move(incumbent->bindingCopyCharge_);
        source_=std::move(incumbent->source_);
        compositionLeaves_=std::move(incumbent->compositionLeaves_);
        parentOwner_=std::move(incumbent->parentOwner_);
        sceneAnimationAdmissionCharge_=std::move(incumbent->sceneAnimationAdmissionCharge_);
        sceneAnimationOwner_=std::move(incumbent->sceneAnimationOwner_);
        sceneAnimationFrames_=std::move(incumbent->sceneAnimationFrames_);
        motion_=std::move(incumbent->motion_); motionOwner_=std::move(incumbent->motionOwner_);
        motionDevice_=std::move(incumbent->motionDevice_); materialScene_=std::move(incumbent->materialScene_);
        lastGoodMaterial_=std::move(incumbent->lastGoodMaterial_);
        highestMaterialRevisionSeen_=incumbent->highestMaterialRevisionSeen_;
        lastGoodMaterialRevision_=incumbent->lastGoodMaterialRevision_;
        rejectedMaterialRevision_=incumbent->rejectedMaterialRevision_;
        // The incumbent copied the epoch before this attempt advanced it.
        compositionEpochPayload_=std::move(incumbent->compositionEpochPayload_);
        compositionEpochIdentity_=incumbent->compositionEpochIdentity_;
        highestCompositionEvaluationRevision_=incumbent->highestCompositionEvaluationRevision_;
    };
    struct Restore final { const decltype(restore)& action; ~Restore() { action(); } } restoreOnExit {restore};
    const bool usesMotion = arbitgpu::sceneUsesMotionPass(request.runtimeInputs);
    const auto rejectMotion = [&](const char* message) {
        motion_[owner == ImportedAnimationEvaluationOwner::Preview ? 0 : 1] = {};
        return reject(error, message);
    };
    const auto rejectMotionWith = [&](const std::string& message) {
        motion_[owner == ImportedAnimationEvaluationOwner::Preview ? 0 : 1] = {};
        error = message;
        return false;
    };
    // The Motion predecessor keeps the authored chain; S5a does not carry it yet.
    if (localTransform && usesMotion)
        return rejectMotion("Imported object local transforms with Motion are not rendered yet");
    if (request.structuralRevision == 0 || request.frame.rateNumerator == 0
        || request.frame.rateDenominator == 0)
        return reject(error, "imported animated scene frame identity is invalid");
    if (request.width == 0 || request.height == 0
        || request.width > 8192 || request.height > 8192
        || static_cast<std::uint64_t>(request.width) * request.height > 67108864u)
        return reject(error, "imported animated scene dimensions exceed the product bound");

    const auto payload = owner == ImportedAnimationEvaluationOwner::Preview
        ? store_.resolvePreview(request.operation.asset)
        : store_.resolveExport(request.operation.asset);
    if (!payload)
        return reject(error, "imported animated scene exact payload is not resident");
    if (composition)
    {
        const std::array<std::uint64_t,5> epochIdentity {request.operation.sourceStableId,
            request.operation.deformationStableId,request.structuralRevision,request.projectGeneration,request.helperGeneration};
        if (compositionEpochPayload_.lock()==payload && compositionEpochIdentity_==epochIdentity)
        {
            if (request.evaluationRevision<highestCompositionEvaluationRevision_)
                return reject(error,"native composition parameter evaluation epoch is stale");
        }
        else highestCompositionEvaluationRevision_=0;
        compositionEpochPayload_=payload;compositionEpochIdentity_=epochIdentity;
        highestCompositionEvaluationRevision_=request.evaluationRevision;
    }
    {
        // Cold admission decodes and adapts the GLB; a warm owner returns at once.
        loopstep::Scope admitScope ("advance:imported-scene:animated-admit");
        if (!admit(payload, request.operation, request.proceduralScene, request.sourceCameraNodeId, request, error)) return false;
    }
    if (request.operation.pose.nodeStableId != 0
        && std::none_of(importedScene_->objects.begin(), importedScene_->objects.begin() + importedScene_->objectCount,
            [&](const auto& object) { return (object.id.value - 1u) / 65536u + 1u
                == request.operation.pose.nodeStableId; }))
        return reject(error, "selected pose object is not drawn by the imported scene selection");
    if (localTransform)
    {
        const auto document = consumer_.admittedDocument();
        const char* refusal = !document || !document->deformation || !baseScene_ || !scene_
            ? "Imported object local transform has no admitted deformation leaf"
            : localTransformRefusal(*baseScene_, *scene_, bindings_, *document->deformation,
                request.objectLocalTransform);
        if (refusal) return reject(error, refusal);
    }

    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> sampleCharge;
    if (composition && bindings_.empty())
    {
        const auto document=consumer_.admittedDocument();
        std::size_t bytes=4*sizeof(visualanimation::Sample)+sizeof(NativeSceneAnimationCompositionBudget::Charge)
            +16*sizeof(std::shared_ptr<const void>);
        const auto add=[&](std::size_t count,std::size_t stride) {
            if (bytes>NativeSceneAnimationCompositionBudget::kMaximumBytes
                || count>(NativeSceneAnimationCompositionBudget::kMaximumBytes-bytes)/stride)
                return reject(error,"camera/light composition sample exceeds logical CPU capacity");
            bytes+=count*stride;return true;
        };
        // Four simultaneous samples cover primary, secondary, crossfade and
        // immutable publication. Key/value scans are admitted before sampling.
        if (!sceneAnimationBudget_->work(document->clips.capacity(),256,error)) return false;
        for (const auto& clip:document->clips)
        {
            if (!clip.clip || !sceneAnimationBudget_->work(clip.clip->tracks().capacity(),256,error)) return false;
            if (!add(clip.clip->tracks().size(),4*sizeof(visualanimation::TrackSample))) return false;
            for (const auto& track:clip.clip->tracks())
                if (!add(track.valueWidth(),4*sizeof(float))
                    || !sceneAnimationBudget_->work(track.keyTimesSeconds().size(),64,error)
                    || !sceneAnimationBudget_->work(track.values().size(),64,error)) return false;
        }
        sampleCharge=NativeSceneAnimationCompositionBudget::reserve(sceneAnimationBudget_,bytes,error);
        if (!sampleCharge) return false;
    }
    ImportedAnimationDeformationEvaluation evaluation;
    bool evaluated = false;
    {
        // Samples the clip and evaluates the skin and morph deformation at this frame.
        loopstep::Scope evaluateScope ("advance:imported-scene:animated-evaluate");
        evaluated = owner == ImportedAnimationEvaluationOwner::Preview
            ? consumer_.evaluatePreview(request.operation, request.frame,
                                        request.structuralRevision, evaluation, error)
            : consumer_.evaluateExport(request.operation, request.frame,
                                       request.structuralRevision, evaluation, error);
    }
    if (!evaluated) return false;
    if (sampleCharge && evaluation.sceneAnimation)
        evaluation.sceneAnimation=retainSceneAnimationAllocation(evaluation.sceneAnimation,sampleCharge);
    if (request.parentIntent && !evaluation.deformation)
        return reject(error,"imported parenting requires an actual skin or morph snapshot");
    const bool sceneAnimationComposition = composition && !evaluation.deformation;
    if (sceneAnimationComposition && (!evaluation.sceneAnimation || !bindings_.empty()))
        return reject(error,"camera/light composition requires an actual no-deformation scene sample");
    bool requiresObjectMotion = false;
    if (usesMotion)
    {
        if (bindings_.empty() || !renderer_.supportsRetainedDeformationMotion())
            return rejectMotion("Motion pass requires the native retained deformation consumer");
        if (request.runtimeInputs.noteInstanceMapping
            && !renderer_.supportsRetainedNoteMotion())
            return rejectMotion("Motion pass requires the native stable-note predecessor consumer");
        const auto document = consumer_.admittedDocument();
        const auto clipId = evaluation.deformation ? evaluation.deformation->clip() : evaluation.sceneAnimation->clip();
        const auto clip = std::find_if(document->clips.begin(), document->clips.end(),
            [&](const auto& value) { return value.clip && value.clip->id() == clipId; });
        if (clip == document->clips.end())
            return rejectMotion("Motion pass has no exact admitted animation clip");
        if (clip->jointBindings.empty() && clip->morphBindings.empty())
            return rejectMotion("Motion pass requires source-bound skin or morph animation");
        // Drawn-node and ancestor animation require exact retained worlds and
        // a consumer capability, checked after source-specific validation.
        std::vector<bool> objectAncestors(baseScene_->nodes.size(), false);
        for (std::size_t index = 0; index < importedScene_->objectCount; ++index)
        {
            std::optional<std::size_t> node = (importedScene_->objects[index].id.value - 1u) / 65536u;
            for (std::size_t visits = 0; node; ++visits)
            {
                if (*node >= baseScene_->nodes.size() || visits >= baseScene_->nodes.size())
                    return rejectMotion("Motion pass has invalid imported object ancestry");
                objectAncestors[*node] = true;
                node = baseScene_->nodes[*node].parent;
            }
        }
        for (const auto& track : clip->clip->tracks())
            if (track.channel() != visualanimation::Channel::MorphWeights
                && track.target().value > 0 && track.target().value <= objectAncestors.size()
                && objectAncestors[track.target().value - 1u])
                requiresObjectMotion = true;
        if (evaluation.deformation)
            for (const auto& transform : evaluation.deformation->jointTransforms())
                if (transform.animationTarget().value > 0
                    && transform.animationTarget().value <= objectAncestors.size()
                    && objectAncestors[transform.animationTarget().value - 1u])
                    requiresObjectMotion = true;
    }

    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> materialCallerCharge;
    if (sceneAnimationComposition && (request.material || !request.surfacePrograms.empty()))
    {
        std::size_t bytes=sizeof(NativeSceneAnimationCompositionBudget::Charge);
        const auto add=[&](std::size_t count,std::size_t stride) {
            if (bytes>NativeSceneAnimationCompositionBudget::kMaximumBytes
                || count>(NativeSceneAnimationCompositionBudget::kMaximumBytes-bytes)/stride)
                return reject(error,"camera/light composition material caller exceeds logical CPU capacity");
            if (!sceneAnimationBudget_->work(count,stride*4,error)) return false;
            bytes+=count*stride;return true;
        };
        const auto material=[&](const surfacematerialbinding::ImportedSceneMaterialRequest& value) {
            if (value.structuralRevision!=request.structuralRevision || value.evaluationRevision!=request.evaluationRevision)
                return reject(error,"camera/light composition material evaluation epoch is stale");
            return add(4,sizeof(value)) && add(value.program.operations.capacity(),4*sizeof(surfacematerial::Operation))
                && add(value.binding.surfaceMaterialDigest.capacity()+1,4)
                && add(value.binding.textures.capacity(),4*sizeof(surfacematerialbinding::TextureSlotBinding))
                && (!value.vertexModifier || add(value.vertexModifier->records.capacity(),4*sizeof(videowire::VertexModifierRecord)));
        };
        if (!add(request.surfacePrograms.capacity(),4*sizeof(geometrysurfacematerial::ObjectProgram))) return false;
        if (request.material && !material(*request.material)) return false;
        for (const auto& object:request.surfacePrograms)
        {
            if (!material(object.program.material)
                || !add(object.program.fields.capacity(),4*sizeof(surfacematerialfield::Binding))) return false;
            for (const auto& field:object.program.fields)
                if (!add(field.sample.fieldPlan.capacity()+1,4)) return false;
        }
        // Existing material admission owns compilation/evaluation and native
        // resources. This distinct debit covers new caller value copies only.
        materialCallerCharge=NativeSceneAnimationCompositionBudget::reserve(sceneAnimationBudget_,bytes,error);
        if (!materialCallerCharge) return false;
    }
    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;
    if (request.material && request.surfacePrograms.empty())
    {
        auto material = *request.material;
        if (material.sceneSnapshot != nullptr)
            return reject(error,
                "imported animated material transport must not carry a foreign scene owner");
        if (proceduralScene_)
        {
            material.scene = scene_->id;
            material.sceneRevision = request.structuralRevision;
        }
        material.sceneSnapshot = scene_;
        if (!publishSurfaceMaterial(material, error))
        {
            if (surfacematerialbinding::hasGraphFrameInput(material.binding))
                return false;
            if (owner != ImportedAnimationEvaluationOwner::Preview
                || lastGoodMaterial_ == nullptr)
                return false;
            usedLastGoodMaterial = true;
            materialDiagnostic = error;
        }
    }
    else if (!request.surfacePrograms.empty())
    {
        if (!proceduralScene_) return reject(error, "Animated Surface collections require their composed procedural owner");
        auto programs = request.surfacePrograms;
        for (auto& object : programs)
        {
            const auto mapped = HarmonicMIDI::grid::scenecomposition::composedProceduralObjectId(object.program.material.binding.object);
            if (!mapped) return reject(error, "Scene Compose procedural Surface target has no object identity");
            object.program.material.binding.object = *mapped;
            object.program.material.sceneSnapshot = scene_;
        }
        if (request.material)
        {
            std::string overrideError;
            if (!geometrysurfacematerial::appendImportedOverride(programs, *request.material,
                    scene_, request.structuralRevision, request.evaluationRevision, overrideError))
                return reject(error, overrideError.c_str());
        }
        videowire::geometry::RuntimeFieldEvaluation fields;
        fields.timelineSeconds = static_cast<double>(request.frame.frame) * request.frame.rateDenominator / request.frame.rateNumerator;
        fields.scoreAt = [&](const auto& operation, const auto& positions, auto& result, auto& diagnostic) {
            return scorefield::evaluateSampleField(operation, positions, request.runtimeInputs.canonicalBlockCFrame, result, diagnostic);
        };
        const auto name = renderer_.backendInfo().backend;
        const auto target = name == "metal" ? videohelper::materialprogram::BackendTarget::Metal
            : name == "opengl" ? videohelper::materialprogram::BackendTarget::OpenGl : videohelper::materialprogram::BackendTarget::Invalid;
        auto admitted = videorender::fixture3d::admitSurfaceMaterialCollection(scene_, programs, fields, target, error);
        if (!admitted) return false;
        materialScene_ = scene_;
        lastGoodMaterial_ = std::move(admitted);
        lastGoodMaterialRevision_ = request.structuralRevision;
        rejectedMaterialRevision_ = 0;
    }
    else
    {
        clearSurfaceMaterial();
    }
    if (owner == ImportedAnimationEvaluationOwner::Export
        && rejectedMaterialRevision_ > lastGoodMaterialRevision_)
        return reject(error,
            "current animated surface material revision is not exportable");

    if (sceneAnimationComposition)
    {
        if (!executeSceneAnimationComposition(owner,request,std::move(evaluation),usedLastGoodMaterial,
                std::move(materialDiagnostic),output,error)) return false;
        // A pending frame publishes nothing, so the attempt is not published.
        if (output.pending) return true;
        published=true;
        return true;
    }

    if (!source_ || source_->sourceStableId != request.operation.sourceStableId
        || source_->deformationStableId != request.operation.deformationStableId
        || source_->structuralRevision != request.structuralRevision
        || (composition && (source_->evaluationRevision!=request.evaluationRevision
            || !source_->compositionOwner || source_->compositionOwner->projectGeneration()!=request.projectGeneration
            || source_->compositionOwner->helperGeneration()!=request.helperGeneration))
        || source_->clip != (evaluation.deformation ? evaluation.deformation->clip() : evaluation.sceneAnimation->clip()))
    {
        std::shared_ptr<const NativeCompositionRetentionBudget::Charge> ownerCharge;
        if (composition)
        {
            const auto n=scene_->objectCount;
            if (n==0 || n>64 || baseScene_->nodes.size()>4096 || bindings_.size()>64)
                return reject(error,"native composition caller cardinalities exceed private owner admission");
            // Original draw assembly and final typed wrappers each copy actual
            // joint/morph vectors. The private owner's own debit is separate.
            std::size_t wrapperBytes=2*sizeof(arbitgpu::NativeDeformationScene)
                +28*sizeof(std::shared_ptr<const void>)+sizeof(NativeCompositionRetentionBudget::Charge)
                +n*(sizeof(arbitgpu::NativeDeformationScene)+sizeof(std::shared_ptr<const arbitgpu::NativeDeformationScene>)
                    +sizeof(arbitgpu::NativeDeformationCompositionMapping))
                +bindings_.size()*(sizeof(arbitgpu::NativeDeformationScene)+sizeof(std::shared_ptr<const arbitgpu::NativeDeformationScene>));
            for (const auto& binding:bindings_)
            {
                if (binding.jointBaseTransforms.size()>64 || binding.morphBaseWeights.size()>8)
                    return reject(error,"native composition caller binding cardinalities exceed private admission");
                wrapperBytes+=3*(binding.jointBaseTransforms.size()*sizeof(arbitgpu::NativeDeformationJointBaseTransform)
                    +binding.morphBaseWeights.size()*sizeof(float));
            }
            // New procedural wrappers copy the factory's one rigid joint.
            wrapperBytes+=proceduralScene_->objectCount*(sizeof(arbitgpu::NativeDeformationJointBaseTransform)
                +4*sizeof(std::shared_ptr<const void>));
            if (wrapperBytes>NativeCompositionRetentionBudget::kCallerMaximumBytes)
                return reject(error,"native composition wrapper storage exceeds its source-derived cap");
            const auto ownerBytes=2*arbitgpu::NativeDeformationCompositionOwner::singleLeafBytes()+4096*sizeof(bool)+12288*sizeof(bool)
                +n*(sizeof(arbitgpu::NativeDeformationScene)+sizeof(visualdeformation::DeformationAsset)
                    +sizeof(visualdeformation::Skin)+sizeof(visualdeformation::Joint)+sizeof(visualdeformation::Mesh)
                    +sizeof(visualdeformation::JointWeightSet)+sizeof(arbitgpu::NativeDeformationJointBaseTransform)
                    +sizeof(std::vector<std::uint32_t>)+sizeof(std::vector<float>))
                +2*4096*4*(sizeof(std::uint32_t)+sizeof(float))+2*n*sizeof(arbitgpu::NativeDeformationFrameData);
            std::size_t inputBytes=0;
            if (!compositionInputBytes(payload,*baseScene_,*consumer_.admittedDocument(),*importedScene_,
                    *proceduralScene_,*scene_,inputBytes,callerLogicalWork,callerWorkLimit,error)) return false;
            // The census prospectively debited two visits per reachable byte
            // before each nested header visit. Copies and mapping have their
            // own named schedules, separate from the private owner work.
            if (!chargeCallerWork(wrapperBytes,2)
                || !chargeCallerWork(n*n,16)) return false;
            if (inputBytes>NativeCompositionRetentionBudget::kMaximumBytes-wrapperBytes
                || ownerBytes>NativeCompositionRetentionBudget::kMaximumBytes-wrapperBytes-inputBytes)
                return reject(error,"native composition full owner inputs exceed logical CPU capacity");
            ownerCharge=NativeCompositionRetentionBudget::reserve(compositionBudget_,wrapperBytes+ownerBytes+inputBytes,error);
            if (!ownerCharge) return false;
        }
        auto source = std::make_shared<arbitgpu::NativeDeformationScene>();
        source->sourceStableId = request.operation.sourceStableId;
        source->deformationStableId = request.operation.deformationStableId;
        source->structuralRevision = request.structuralRevision;
        source->clip = evaluation.deformation ? evaluation.deformation->clip() : evaluation.sceneAnimation->clip();
        source->scene = composition ? importedScene_ : scene_;
        if (composition)
        {
            source->evaluationRevision=request.evaluationRevision;source->logicalCompositionRetention=ownerCharge;
            source->scene=retainCompositionAllocation(importedScene_,ownerCharge);
        }
        source->deformation = consumer_.admittedDocument()->deformation;
        if (composition) source->deformation=retainCompositionAllocation(source->deformation,ownerCharge);
        if (composition) source->draws.reserve(bindings_.size());
        for (const auto& binding : bindings_)
        {
            auto draw = composition ? std::make_shared<arbitgpu::NativeDeformationScene>()
                                    : std::make_shared<arbitgpu::NativeDeformationScene>(*source);
            if (composition)
            {
                draw->sourceStableId=source->sourceStableId;draw->deformationStableId=source->deformationStableId;
                draw->structuralRevision=source->structuralRevision;draw->evaluationRevision=source->evaluationRevision;
                draw->clip=source->clip;draw->scene=source->scene;draw->deformation=source->deformation;
                draw->logicalCompositionRetention=ownerCharge;
                draw->jointBaseTransforms.reserve(binding.jointBaseTransforms.size());
            }
            draw->draws.clear();
            draw->mesh = binding.mesh;
            draw->skin = binding.skin;
            draw->animationNodeStableId = binding.nodeIndex + 1u;
            draw->object = {static_cast<std::uint32_t>(binding.nodeIndex * 65536u + 1u)};
            draw->morphBaseWeights = binding.morphBaseWeights;
            draw->batchMember = true;
            for (const auto& base : binding.jointBaseTransforms)
                draw->jointBaseTransforms.push_back({base.skin, base.joint,
                    base.translation, base.rotation, base.scale, base.matrix});
            source->draws.push_back(std::move(draw));
        }
        if (!source->draws.empty())
        {
            source->mesh = source->draws.front()->mesh;
            source->object = source->draws.front()->object;
        }
        if (!composition && source->draws.size() == 1 && scene_->objectCount == 1
            && scene_->materialCount == 1 && scene_->cameraCount == 1 && scene_->lightCount <= 1
            && scene_->materials[0].opacity == 1.0f
            && (scene_->lightCount == 0
                || scene_->lights[0].kind == HarmonicMIDI::grid::SceneLightKind::Directional))
        {
            const auto draw = source->draws.front();
            *source = *draw;
            source->batchMember = false;
        }
        if (composition)
        {
            std::vector<arbitgpu::NativeDeformationCompositionMapping> mappings;
            mappings.reserve(scene_->objectCount);
            for (std::size_t i=0;i<importedScene_->objectCount;++i)
            {
                const auto id=importedScene_->objects[i].id;
                mappings.push_back({arbitgpu::NativeDeformationCompositionMapping::Kind::Imported,
                    id,id,(id.value-1u)/65536u+1u});
            }
            for (std::size_t i=0;i<proceduralScene_->objectCount;++i)
            {
                const auto id=proceduralScene_->objects[i].id;
                const auto target=HarmonicMIDI::grid::scenecomposition::composedProceduralObjectId(id);
                if (!target) return reject(error,"native composition procedural mapping has no exact composed identity");
                mappings.push_back({arbitgpu::NativeDeformationCompositionMapping::Kind::Procedural,id,*target,0});
            }
            const auto original=retainCompositionAllocation<arbitgpu::NativeDeformationScene>(source,ownerCharge);
            const auto retainedPayload=retainCompositionAllocation(parentOwner_?parentOwner_->payload_:payload,ownerCharge);
            const auto retainedProcedural=retainCompositionAllocation(parentOwner_?parentOwner_->procedural_:proceduralScene_,ownerCharge);
            const auto retainedComposed=retainCompositionAllocation(scene_,ownerCharge);
            auto owner=arbitgpu::NativeDeformationCompositionOwner::admit(retainedPayload,baseScene_,consumer_.admittedDocument(),
                request.operation.sceneIndex,request.operation.meshStableId,original,retainedProcedural,retainedComposed,mappings,
                request.projectGeneration,request.helperGeneration,error,request.evaluationRevision);
            if (!owner) return false;
            if (parentOwner_)
            {
                auto issuing=std::const_pointer_cast<ImportedParentBindingOwner>(parentOwner_);
                const auto charge=std::static_pointer_cast<const NativeCompositionRetentionBudget::Charge>(issuing->retention_);
                owner=retainParentAllocation(owner,charge,sceneAnimationBudget_);
                issuing->native_=owner;
                if (!issuing->inspect(*request.parentIntent))
                    return reject(error,"imported parent native owner failed independent conservation");
            }
            auto batch=std::make_shared<arbitgpu::NativeDeformationScene>();
            batch->sourceStableId=source->sourceStableId;batch->deformationStableId=source->deformationStableId;
            batch->structuralRevision=source->structuralRevision;batch->evaluationRevision=request.evaluationRevision;
            batch->clip=source->clip;batch->scene=owner->scene();batch->deformation=source->deformation;
            batch->compositionOwner=owner;batch->draws.reserve(owner->memberCount());
            for (std::size_t i=0;i<owner->memberCount();++i)
            {
                auto draw=std::make_shared<arbitgpu::NativeDeformationScene>(*owner->originalMember(i));
                draw->scene=owner->scene();draw->object=owner->memberObject(i);draw->compositionOwner=owner;
                draw->draws.clear();draw->batchMember=true;
                batch->draws.push_back(std::move(draw));
            }
            if (batch->draws.empty()) return reject(error,"native composition has no complete native members");
            batch->mesh=batch->draws.front()->mesh;batch->object=batch->draws.front()->object;
            source_=retainCompositionAllocation<arbitgpu::NativeDeformationScene>(std::move(batch),ownerCharge);
        }
        else source_ = std::move(source);
    }

    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> selectedCameraCharge;
    if (request.sourceCameraNodeId && evaluation.deformation)
    {
        // Five escaped aliases each store value and charge. Caller staging also
        // retains the charge, selected frame, issued frame, payload and receipt
        // field. Each alias invocation can simultaneously hold its value
        // parameter, two-field Retained temporary, retained local and return
        // temporary. Count all five invocations even without overlap/elision.
        constexpr auto aliases=5u, callerPointers=5u, aliasStagingPointers=5u;
        constexpr auto bytes=sizeof(NativeSelectedDeformationCameraFrame)
            +sizeof(NativeSceneAnimationCompositionBudget::Charge)
            +(2*aliases+callerPointers+aliases*aliasStagingPointers)*sizeof(std::shared_ptr<const void>);
        // Eight byte-lane passes bound initialization, camera/override capture,
        // alias staging/copies and final publication. Native/allocator internals
        // remain separate from this existing logical CPU allowance.
        if (!sceneAnimationBudget_->work(bytes,8,error)) return false;
        selectedCameraCharge=NativeSceneAnimationCompositionBudget::reserve(sceneAnimationBudget_,bytes,error);
        if (!selectedCameraCharge) return false;
    }

    std::shared_ptr<const arbitgpu::NativeDeformationCompositionFrame> compositionFrame;
    std::shared_ptr<const NativeCompositionRetentionBudget::Charge> frameCharge;
    if (composition)
    {
        std::size_t bytes=0;
        if (!chargeCallerWork(evaluation.deformation->morphWeights().size(),32)
            || (evaluation.sceneAnimation && !chargeCallerWork(evaluation.sceneAnimation->tracks().size(),32))) return false;
        if (!compositionFrameBytes(*source_->compositionOwner,evaluation,bytes,error)) return false;
        if (request.sourceCameraNodeId && evaluation.sceneAnimation)
        {
            const auto nodes=baseScene_->nodes.size();
            const auto records=importedScene_->cameraCount+importedScene_->lightCount;
            if (nodes>16384 || records>HarmonicMIDI::grid::Visual3DScene::kMaxCameras+HarmonicMIDI::grid::Visual3DScene::kMaxLights
                || !chargeCallerWork(nodes,256) || !chargeCallerWork(records*nodes,64)
                || !chargeCallerWork(evaluation.sceneAnimation->tracks().size(),128)) return false;
            const auto sampleBytes=nodes*(3*sizeof(std::array<float,16>)+2)
                +2*(HarmonicMIDI::grid::Visual3DScene::kMaxCameras*sizeof(HarmonicMIDI::grid::SceneCameraRecord)
                    +HarmonicMIDI::grid::Visual3DScene::kMaxLights*sizeof(HarmonicMIDI::grid::SceneLightRecord));
            if (bytes>NativeCompositionRetentionBudget::kMaximumBytes
                || sampleBytes>NativeCompositionRetentionBudget::kMaximumBytes-bytes)
                return reject(error,"selected camera composition sampling exceeds logical CPU capacity");
            bytes+=sampleBytes;
        }
        if (request.runtimeInputs.noteInstanceMapping && request.runtimeInputs.canonicalBlockCFrame)
        {
            const auto noteBytes=2*sizeof(arbitgpu::NativeNoteInstanceBatch)+sizeof(arbitgpu::NativeSceneNoteMotionTransforms)
                +4*sizeof(std::shared_ptr<const void>);
            if (noteBytes>NativeCompositionRetentionBudget::kMaximumBytes-bytes)
                return reject(error,"native composition note Motion storage exceeds logical CPU capacity");
            bytes+=noteBytes;
            const auto rows=std::min<std::size_t>(request.runtimeInputs.canonicalBlockCFrame->noteRows(),visualnoteinstancing::kMaximumInstances);
            if (!chargeCallerWork(rows,128) || !chargeCallerWork(noteBytes,2)) return false;
        }
        frameCharge=NativeCompositionRetentionBudget::reserve(compositionBudget_,bytes,error);
        if (!frameCharge) return false;
        evaluation.deformation=retainCompositionAllocation(evaluation.deformation,frameCharge);
        if (evaluation.sceneAnimation) evaluation.sceneAnimation=retainCompositionAllocation(evaluation.sceneAnimation,frameCharge);
        compositionFrame=arbitgpu::NativeDeformationCompositionFrame::admit(source_->compositionOwner,
            evaluation.deformation,evaluation.sceneAnimation,request.frame,request.projectGeneration,request.helperGeneration,error);
        if (!compositionFrame) return false;
    }

    videorender::animation3d::RenderedDeformationFrame frame;
    const auto materialScene = materialScene_.lock();
    const auto material = materialScene == scene_ ? lastGoodMaterial_ : nullptr;
    const auto program = material ? material->nativeProgram() : nullptr;
    const bool hasVertexProgram = program && (program->vertexProgram
            || std::any_of(program->objectPrograms.begin(), program->objectPrograms.end(),
                [](const auto& value) { return value && value->vertexProgram; }));
    if (request.parentIntent && hasVertexProgram)
        return reject(error,"imported parenting does not admit a material vertex processor");
    // The native draw runs a vertex program again at the prior time, spectrum
    // and prior skinned or morphed position. Normals are not kept for that pose.
    if (usesMotion && hasVertexProgram)
    {
        const auto refusal = [](const auto& value) {
            return value && value->vertexProgram ? videowire::vertexModifierPriorGeometryRefusal(
                value->vertexProgram->vertexInstructions(), videowire::VertexModifierPriorGeometry::skinnedPose) : nullptr;
        };
        const char* message = refusal(program);
        for (const auto& value : program->objectPrograms)
            if (!message) message = refusal(value);
        if (message) return rejectMotion(message);
    }
    const auto exactSeconds = static_cast<double>(request.frame.frame)
        * static_cast<double>(request.frame.rateDenominator)
        / static_cast<double>(request.frame.rateNumerator);
    if (!std::isfinite(exactSeconds)
        || std::abs(exactSeconds) > surfacematerial::kMaximumEvaluationMagnitude)
        return reject(error, "imported animated material time is out of bounds");
    if (request.runtimeInputs.lightOverride && !request.importedLights.intensityMultipliers.empty())
        return reject(error, "Render 3D Light input replaces imported lighting");
    auto runtimeInputs = request.runtimeInputs;
    if (request.sourceCameraNodeId && proceduralScene_ && runtimeInputs.cameraOverride)
    {
        using namespace HarmonicMIDI::grid;
        if (!chargeCallerWork(proceduralScene_->cameraCount+scene_->cameraCount,sizeof(SceneCameraRecord))) return false;
        const auto* authored=visual3d_detail::findById(proceduralScene_->cameras,
            proceduralScene_->cameraCount,proceduralScene_->activeCamera);
        const auto* retained=visual3d_detail::findById(scene_->cameras,scene_->cameraCount,scene_->activeCamera);
        if (!authored || !retained || !scenecomposition::sameCameraRecord(*authored,*runtimeInputs.cameraOverride)
            || !scenecomposition::sameCameraRecord(*authored,*retained))
            return reject(error,"composed deformation camera override does not match its exact authored and retained records");
        runtimeInputs.cameraOverride=*retained;
    }
    if (request.sourceCameraNodeId) runtimeInputs.animatedCamera.reset();
    if (composition) runtimeInputs.compositionFrame=compositionFrame;
    if (proceduralScene_ && !runtimeInputs.noteInstanceMapping) runtimeInputs.canonicalBlockCFrame.reset();
    if (evaluation.sceneAnimation && (!proceduralScene_ || request.sourceCameraNodeId))
    {
        std::vector<HarmonicMIDI::grid::SceneCameraRecord> cameras;
        std::vector<HarmonicMIDI::grid::SceneLightRecord> compositionLights;
        auto& sampledLights = proceduralScene_ ? compositionLights : runtimeInputs.animatedLights;
        if (!gltf::sampleGlbCameraLights(*baseScene_, *evaluation.sceneAnimation, request.operation.combinationMode,
                static_cast<float>(evaluation.playback.weight), *importedScene_, cameras, sampledLights, error))
            return false;
        const auto camera = std::find_if(cameras.begin(), cameras.end(),
            [&](const auto& value) { return request.sourceCameraNodeId ? value.id.value == *request.sourceCameraNodeId : value.id == scene_->activeCamera; });
        if (request.sourceCameraNodeId && camera == cameras.end())
            return reject(error, "selected imported camera has no exact sampled record");
        if (camera != cameras.end()) runtimeInputs.animatedCamera = *camera;
    }
    if (request.sourceCameraNodeId && !runtimeInputs.animatedCamera)
    {
        const auto* camera = sourceCameraRecord(*baseScene_, *importedScene_, *request.sourceCameraNodeId);
        if (!camera) return reject(error, "selected imported camera owner is stale");
        runtimeInputs.animatedCamera = *camera;
    }
    if (!proceduralScene_ && !importedscene::evaluateImportedLights(request.importedLights,
            importedscene::importedLightDefinitions(*baseScene_, *importedScene_), *importedScene_,
            runtimeInputs.animatedLights, error)) return false;
    // Overrides stay unapplied here. The backend reader applies them once, to the
    // record each light index reads, so offsets are never applied twice.
    runtimeInputs.objectNodeStableId = request.operation.pose.nodeStableId;
    runtimeInputs.timeSeconds = static_cast<float>(exactSeconds);
    runtimeInputs.morphWeight = request.operation.pose.morphEnabled || request.operation.combinationMode
            == visualanimation::CombinationMode::WeightedBlend
        ? 1.0f : static_cast<float>(request.operation.playback.weight);
    if (!arbitgpu::sceneMotionSnapshotFits(evaluation.deformation))
        return reject(error, "imported animated Motion predecessor exceeds the retained snapshot bound");
    if (evaluation.deformation && !(evaluation.deformation->time() == request.frame))
        return reject(error, "imported animated snapshot presentation frame disagrees with the draw");
    const arbitgpu::NativeSceneMotionOwner motionOwner {{ source_->sourceStableId,
        source_->deformationStableId, source_->clip.value, source_->structuralRevision,
        request.projectGeneration, request.helperGeneration,
        static_cast<std::uint64_t>(request.renderClipId), runtimeInputs.objectNodeStableId },
        composition ? request.evaluationRevision : 0 };
    const auto device = renderer_.backendInfo();
    const auto motionDevice = device.backend + "\n" + device.device;
    // The history reset waits for a render that publishes. Until then the
    // predecessor reads as it would after the reset, and nothing is written.
    const bool motionHistoryReset = (motionOwner_ && !(*motionOwner_ == motionOwner)) || motionDevice_ != motionDevice;
    auto& motion = motion_[owner == ImportedAnimationEvaluationOwner::Preview ? 0 : 1];
    if (motionHistoryReset) runtimeInputs.previousMotion = std::nullopt;
    else runtimeInputs.previousMotion = motion.predecessor(request.frame.frame,
        request.frame.rateNumerator, request.frame.rateDenominator,
        request.width, request.height, motionOwner, runtimeInputs);
    // The private frame schedule includes its native current/previous dispatch.
    // Four additional typed CPU prepares cover current capture, predecessor
    // revalidation and renderer preflight. Caller scans have independent work.
    if (compositionFrame)
    {
        const auto nodes=baseScene_->nodes.size(),objects=importedScene_->objectCount;
        // Two world captures and, when selected, one Motion ancestry scan.
        // Each ancestor visit has at most 32 source tests/reads/updates. Each
        // node-table visit covers 16 finite lanes, table addressing, copying
        // and publication with 256 source steps. Track/joint identity visits
        // reserve 64 steps through captured() and selected-ancestor checks.
        const auto scanWork=(2+(usesMotion?1u:0u))*objects*nodes*32+2*nodes*256
            +3*evaluation.deformation->jointTransforms().size()*64
            +(evaluation.sceneAnimation?3*evaluation.sceneAnimation->tracks().size()*64:0);
        const auto work=compositionFrame->logicalWork();
        const auto previousWork=runtimeInputs.previousMotion && runtimeInputs.previousMotion->compositionFrame
            ?runtimeInputs.previousMotion->compositionFrame->logicalWork():0;
        constexpr auto limit=callerWorkLimit;
        // Current factory and two current prepares, two predecessor prepares,
        // plus the authenticated owner's private work. The dispatch reservation
        // in the private frame is intentionally counted again as an upper bound.
        const auto preparationWork=3*work+2*previousWork+source_->compositionOwner->logicalWork();
        if (preparationWork>limit || scanWork>limit-preparationWork
            || callerLogicalWork>limit-preparationWork-scanWork)
            return reject(error,"native composition caller capture and validation work exceed capacity");
    }
    std::string nodeWorldFailure;
    auto nodeWorlds = importedNodeWorlds(*baseScene_, *source_, *importedScene_, evaluation,compositionFrame,&nodeWorldFailure);
    if (compositionFrame && nodeWorlds) nodeWorlds=retainCompositionAllocation(nodeWorlds,frameCharge);
    if (localTransform)
    {
        // An affected drawn object, the moved node or one below it, is drawn at C * M,
        // with C = W L W^-1 for the moved node's animated world W at this frame and M
        // its authored chain. Its palette applies inverse(M) * W, so the raster returns
        // W L v. Unaffected drawn objects keep their authored chain.
        const std::array<float, 16>* movedWorld = nullptr;
        if (nodeWorlds)
            for (const auto& entry : *nodeWorlds)
                if (entry.nodeStableId == request.objectLocalTransform.nodeStableId) movedWorld = &entry.world;
        if (!movedWorld)
        {
            error = "Imported object local transform has no exact animated node world";
            if (!nodeWorldFailure.empty()) error += ": " + nodeWorldFailure;
            return false;
        }
        const auto& local = request.objectLocalTransform;
        HarmonicMIDI::grid::SceneTransform3D offset;
        offset.translation = {float(local.translation[0]), float(local.translation[1]), float(local.translation[2])};
        offset.rotation = arbitgpu::runtimeEulerQuaternion({float(local.rotationDegrees[0]), float(local.rotationDegrees[1]), float(local.rotationDegrees[2])});
        offset.scale = {float(local.scale[0]), float(local.scale[1]), float(local.scale[2])};
        NativeSceneAnimationCompositionFrame::Matrix inverseMoved {};
        if (!NativeSceneAnimationCompositionFrame::inverseAffine(*movedWorld, inverseMoved))
            return reject(error, "Imported object local transform animated world is singular or nonfinite");
        const auto correction = NativeSceneAnimationCompositionFrame::multiply(
            NativeSceneAnimationCompositionFrame::multiply(*movedWorld, NativeSceneAnimationCompositionFrame::compose(offset)),
            inverseMoved);
        auto worlds = std::make_shared<arbitgpu::NativeImportedObjectWorlds>();
        for (std::size_t index = 0; index < scene_->objectCount; ++index)
        {
            const auto& object = scene_->objects[index];
            if (!object.indexCount) continue;
            NativeSceneAnimationCompositionFrame::Matrix authored {};
            if (!NativeSceneAnimationCompositionFrame::originalWorld(*scene_, object.id, authored))
                return reject(error, "Imported object local transform draw identity is invalid");
            const auto node = static_cast<std::size_t>((object.id.value - 1u) / 65536u);
            const auto affects = node < baseScene_->nodes.size()
                ? localTransformAffects(*baseScene_, node, request.objectLocalTransform.nodeStableId) : std::nullopt;
            if (!affects) return reject(error, "Imported object local transform draw ancestry is invalid");
            worlds->push_back({object.id, *affects
                ? NativeSceneAnimationCompositionFrame::multiply(correction, authored) : authored});
        }
        if (!arbitgpu::validNativeImportedObjectWorlds(*worlds)
            || !arbitgpu::nativeImportedObjectWorldsExactCover(*scene_, *worlds))
            return reject(error, "Imported object local transform worlds do not name every drawn object exactly");
        runtimeInputs.objectWorlds = std::move(worlds);
        // Motion is refused above, so no predecessor may carry another object set.
        runtimeInputs.previousMotion.reset();
    }
    if (runtimeInputs.previousMotion)
    {
        const auto& previous = *runtimeInputs.previousMotion;
        if (!arbitgpu::validSceneMotionImportedNodeWorldMatrices(previous, requiresObjectMotion))
            return rejectMotion("imported Motion node world table is malformed or exceeds the snapshot bound");
        if (previous.importedNodeWorldMatrices || requiresObjectMotion)
        {
            if (!previous.importedNodeWorldMatrices
                || ((requiresObjectMotion || evaluation.deformation) && !previous.deformation)
                || !(previous.owner == motionOwner)
                || request.frame.frame == std::numeric_limits<std::int64_t>::min()
                || (previous.deformation && !(previous.deformation->time() == visualdeformation::RationalFrameTime {
                    request.frame.frame - 1, request.frame.rateNumerator, request.frame.rateDenominator })))
                return rejectMotion("imported Motion node worlds have no exact predecessor snapshot");
            auto previousEvaluation = evaluation;
            previousEvaluation.deformation = previous.deformation;
            std::string expectedFailure;
            const auto expected = importedNodeWorlds(*baseScene_, *source_, *importedScene_, previousEvaluation,previous.compositionFrame,&expectedFailure);
            const auto& retained = *previous.importedNodeWorldMatrices;
            if (!expected || expected->size() != retained.size()
                || !std::equal(expected->begin(), expected->end(), retained.begin(),
                    [](const auto& canonical, const auto& entry) {
                        return canonical.nodeStableId == entry.nodeStableId && canonical.world == entry.world;
                    }))
                return rejectMotionWith(expectedFailure.empty()
                    ? std::string("imported Motion node worlds disagree with the exact source snapshot")
                    : "imported Motion node worlds disagree with the exact source snapshot: " + expectedFailure);
        }
    }
    if (requiresObjectMotion)
    {
        if (!nodeWorlds)
            return rejectMotion("Motion pass requires complete canonical imported object matrices");
        if (!renderer_.supportsRetainedImportedObjectMotion())
            return rejectMotion("Motion pass requires the native retained imported object matrices consumer");
    }
    bool renderPending = false;
    const auto renderFrame = [&]() {
        loopstep::Scope renderScope (owner == ImportedAnimationEvaluationOwner::Preview
            ? "advance:imported-scene:animated-render" : "advance:imported-scene:animated-export-render");
        if (bindings_.empty())
        {
            videorender::fixture3d::RenderedFrame sceneFrame;
            surfacematerial::MaterialEvaluationInputs materialInputs;
            materialInputs.timeSeconds = runtimeInputs.timeSeconds;
            const auto rendered = owner == ImportedAnimationEvaluationOwner::Preview
                ? sceneRenderer_.renderPreview(scene_, material, materialInputs, runtimeInputs,
                    {request.width, request.height}, videorender::animation3d::kNativeGpuCapability, sceneFrame, error)
                : sceneRenderer_.renderExport(scene_, material, materialInputs, runtimeInputs,
                    {request.width, request.height}, videorender::animation3d::kNativeGpuCapability, sceneFrame, error);
            if (!rendered) return false;
            if (sceneFrame.pending)
            {
                // Export never sees pending, so it refuses instead of publishing nothing.
                if (owner != ImportedAnimationEvaluationOwner::Preview)
                    return reject(error, "imported animated export render returned pending");
                renderPending = true;
                return true;
            }
            frame.use = owner == ImportedAnimationEvaluationOwner::Preview
                ? videorender::animation3d::RenderUse::Preview : videorender::animation3d::RenderUse::Export;
            frame.source = source_; frame.nativeFrame = std::move(sceneFrame.nativeFrame);
            frame.stats.drawCount = sceneFrame.stats.drawCount;
            frame.stats.clipId = source_->clip.value; frame.stats.time = request.frame;
            frame.stats.sourceStableId = source_->sourceStableId;
            frame.stats.deformationStableId = source_->deformationStableId;
            frame.stats.revision = request.structuralRevision;
            frame.stats.staticUploadCount = sceneFrame.stats.staticUploadCount;
            frame.stats.reusedStaticResources = sceneFrame.stats.reusedStaticResources;
            return true;
        }
        return owner == ImportedAnimationEvaluationOwner::Preview
        ? renderer_.renderPreview(source_, evaluation.deformation,
                                  material ? material->nativeProgram() : nullptr,
                                  runtimeInputs, request.width, request.height,
                                  videorender::animation3d::kNativeGpuCapability, frame, error)
        : renderer_.renderExport(source_, evaluation.deformation,
                                 material ? material->nativeProgram() : nullptr,
                                 runtimeInputs, request.width, request.height,
                                 videorender::animation3d::kNativeGpuCapability, frame, error);
    };
    std::shared_ptr<const NativeCompositionRetentionBudget::Charge> parentFrameCharge;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> parentRenderedSource;
    if (request.parentIntent)
    {
        constexpr auto aliases=7u;
        std::size_t bytes=sizeof(ImportedParentBindingFrame)+sizeof(ImportedAnimatedSceneReceipt)
            +sizeof(arbitgpu::NativeSceneMotionSample)+sizeof(NativeCompositionRetentionBudget::Charge)
            +aliases*10*sizeof(std::shared_ptr<const void>);
        for (const auto* text:{&request.operation.asset.id,&request.operation.asset.contentSha256,&request.operation.asset.sourceMediaType,
                              &request.operation.clipName,&request.operation.crossfadeClipName})
        {
            if (text->capacity()>=NativeCompositionRetentionBudget::kMaximumBytes
                || text->capacity()+1>NativeCompositionRetentionBudget::kMaximumBytes-bytes)
                return reject(error,"imported parent frame selection capacity exceeds logical CPU capacity");
            bytes+=text->capacity()+1;
        }
        // The parent frame keeps a copy of the exact light records it renders
        // with (matches compares them). Four identity copies cover stored,
        // runtime, expected-request and staging lanes.
        for (const auto& light:runtimeInputs.animatedLights)
        {
            if (!NativeSceneAnimationCompositionBudget::addLightIdentityBytes(light,4,bytes,error)
                || !sceneAnimationBudget_->consumeLightIdentityWork(light,4))
                return reject(error,"imported parent frame light identity exceeds its reserved capacity");
        }
        if (runtimeInputs.animatedLights.size()>HarmonicMIDI::grid::Visual3DScene::kMaxLights
            || bytes>NativeCompositionRetentionBudget::kMaximumBytes-runtimeInputs.animatedLights.size()*sizeof(HarmonicMIDI::grid::SceneLightRecord))
            return reject(error,"imported parent frame light records exceed logical CPU capacity");
        bytes+=runtimeInputs.animatedLights.size()*sizeof(HarmonicMIDI::grid::SceneLightRecord);
        // Environment light and modulation overrides are stored and compared with the frame.
        bytes+=sizeof(std::optional<HarmonicMIDI::grid::SceneLightRecord>);
        if (runtimeInputs.lightOverride)
        {
            if (!NativeSceneAnimationCompositionBudget::addLightIdentityBytes(*runtimeInputs.lightOverride,4,bytes,error)
                || !sceneAnimationBudget_->consumeLightIdentityWork(*runtimeInputs.lightOverride,4))
                return reject(error,"imported parent frame light override identity exceeds its reserved capacity");
        }
        if (runtimeInputs.lightOverrides.size()>HarmonicMIDI::grid::Visual3DScene::kMaxLights
            || bytes>NativeCompositionRetentionBudget::kMaximumBytes
                -runtimeInputs.lightOverrides.size()*sizeof(arbitgpu::NativeImportedLightOverride))
            return reject(error,"imported parent frame light overrides exceed logical CPU capacity");
        bytes+=runtimeInputs.lightOverrides.size()*sizeof(arbitgpu::NativeImportedLightOverride);
        if (!sceneAnimationBudget_->work(bytes,2,error) || !parentOwner_ || !parentOwner_->inspect(*request.parentIntent))
            return reject(error,"imported parent pre-submit association is missing or exhausted");
        parentFrameCharge=NativeCompositionRetentionBudget::reserve(compositionBudget_,bytes,error);
        if (!parentFrameCharge) return false;
        compositionFrame=retainParentAllocation(compositionFrame,parentFrameCharge,sceneAnimationBudget_);
        runtimeInputs.compositionFrame=compositionFrame;
        // Per-frame aliases start from the stable source. Updating source_ with
        // its own wrapper would retain every past frame debit indefinitely.
        parentRenderedSource=retainParentAllocation(source_,parentFrameCharge,sceneAnimationBudget_);
        evaluation.deformation=retainParentAllocation(evaluation.deformation,parentFrameCharge,sceneAnimationBudget_);
        if (evaluation.sceneAnimation) evaluation.sceneAnimation=retainParentAllocation(evaluation.sceneAnimation,parentFrameCharge,sceneAnimationBudget_);
        if (nodeWorlds) nodeWorlds=retainParentAllocation(nodeWorlds,parentFrameCharge,sceneAnimationBudget_);
    }
    if (parentStaging && !parentStaging->prepareDiagnostic(materialDiagnostic,error)) return false;
    if (!renderFrame()) return false;
    if (renderPending)
    {
        // Retry state. No motion history is reset or committed, no frame is published,
        // and the caller keeps its receipt.
        output.pending = true;
        error.clear();
        return true;
    }
    if (motionHistoryReset) motion_ = {};
    motionOwner_ = motionOwner; motionDevice_ = motionDevice;
    // Samples keep the vertex program time and spectrum, so a graph vertex
    // program shares this history with the retained pose.
    if (!motion.commit(request.frame.frame, request.frame.rateNumerator, request.frame.rateDenominator,
            request.width, request.height, runtimeInputs, evaluation.deformation, motionOwner))
        motion = {};
    else
    {
        motion.current->importedNodeWorldMatrices = nodeWorlds;
        if (compositionFrame && motion.current->noteTransforms)
            motion.current->noteTransforms=retainCompositionAllocation(motion.current->noteTransforms,frameCharge);
    }

    std::shared_ptr<const NativeSelectedDeformationCameraFrame> selectedCamera;
    if (selectedCameraCharge)
    {
        // These aliases preserve the debit if any actual owner escapes without
        // the receipt. Their pointed-to native/source/sample identities stay exact.
        const auto selectedPayload=composition ? source_->compositionOwner->payload_ : payload;
        auto issued=std::shared_ptr<NativeSelectedDeformationCameraFrame>(new NativeSelectedDeformationCameraFrame);
        issued->charge_=selectedCameraCharge;
        issued->payload_=retainSceneAnimationAllocation(selectedPayload,selectedCameraCharge);
        issued->source_=retainSceneAnimationAllocation(source_,selectedCameraCharge);
        evaluation.deformation=retainSceneAnimationAllocation(evaluation.deformation,selectedCameraCharge);
        if (evaluation.sceneAnimation)
            evaluation.sceneAnimation=retainSceneAnimationAllocation(evaluation.sceneAnimation,selectedCameraCharge);
        frame.nativeFrame=retainSceneAnimationAllocation(frame.nativeFrame,selectedCameraCharge);
        frame.deformation=evaluation.deformation;
        issued->deformation_=evaluation.deformation;issued->sample_=evaluation.sceneAnimation;
        issued->nativeFrame_=frame.nativeFrame;issued->camera_=*runtimeInputs.animatedCamera;
        issued->override_=runtimeInputs.cameraOverride;issued->selector_=*request.sourceCameraNodeId;
        issued->use_=owner;issued->time_=request.frame;issued->timeSeconds_=runtimeInputs.timeSeconds;
        issued->width_=request.width;issued->height_=request.height;
        selectedCamera=std::move(issued);
    }

    std::shared_ptr<const ImportedParentBindingFrame> parentBinding;
    if (request.parentIntent)
    {
        auto issued=std::shared_ptr<ImportedParentBindingFrame>(new ImportedParentBindingFrame);
        issued->retention_=parentFrameCharge;issued->owner_=parentOwner_;
        issued->payload_=retainParentAllocation(parentOwner_->payload_,parentFrameCharge,sceneAnimationBudget_);
        issued->source_=parentRenderedSource;issued->composition_=compositionFrame;
        issued->deformation_=evaluation.deformation;issued->sample_=evaluation.sceneAnimation;
        frame.nativeFrame=retainParentAllocation(frame.nativeFrame,parentFrameCharge,sceneAnimationBudget_);
        issued->nativeFrame_=frame.nativeFrame;issued->camera_=runtimeInputs.animatedCamera;
        issued->override_=runtimeInputs.cameraOverride;issued->cameraSelector_=request.sourceCameraNodeId;
        issued->previous_=runtimeInputs.previousMotion;issued->use_=owner;issued->time_=request.frame;
        issued->width_=request.width;issued->height_=request.height;issued->timeSeconds_=runtimeInputs.timeSeconds;
        issued->node_=runtimeInputs.objectNodeStableId;issued->imageOutput_=runtimeInputs.imageOutput;
        issued->operation_=request.operation;issued->playback_=evaluation.playback;
        if (!sceneAnimationBudget_->work(1,10,error) || !ImportedParentBindingFrame::capacitiesFit(issued->operation_,request.operation))
            return reject(error,"imported parent copied frame identity exceeds its reserved capacities");
        issued->rawExportMask_=runtimeInputs.rawExportMask;issued->linearColor_=runtimeInputs.linearColor;
        // Charged above with the parent frame; matches compares these records.
        issued->lights_=runtimeInputs.animatedLights;
        issued->lightOverride_=runtimeInputs.lightOverride;issued->overrides_=runtimeInputs.lightOverrides;
        issued->translation_=runtimeInputs.objectTranslationOffset;issued->rotation_=runtimeInputs.objectRotationDegrees;
        issued->scale_=runtimeInputs.objectScale;issued->cameraTranslation_=runtimeInputs.cameraTranslationOffset;
        issued->cameraRotation_=runtimeInputs.cameraRotationDegrees;issued->cameraFov_=runtimeInputs.cameraVerticalFovDegrees;
        issued->morphWeight_=runtimeInputs.morphWeight;issued->emissionGain_=runtimeInputs.emissionGain;
        frame.source=parentRenderedSource;frame.deformation=evaluation.deformation;
        parentBinding=std::move(issued);
    }
    ImportedAnimatedSceneReceipt receipt;
    receipt.parentBinding_=std::move(parentBinding);
    receipt.selectedDeformationCamera_=std::move(selectedCamera);
    receipt.owner = owner;
    receipt.sourceCameraNodeId = request.sourceCameraNodeId;
    receipt.payload = receipt.selectedDeformationCamera_ ? receipt.selectedDeformationCamera_->payload_
        : composition ? source_->compositionOwner->payload_ : payload;
    receipt.source = receipt.selectedDeformationCamera_ ? receipt.selectedDeformationCamera_->source_
        : receipt.parentBinding_ ? receipt.parentBinding_->source_ : source_;
    if (receipt.selectedDeformationCamera_) frame.source=receipt.source;
    receipt.evaluation = std::move(evaluation);
    receipt.frame = std::move(frame);
    receipt.runtimeInputs = std::move(runtimeInputs);
    receipt.usedLastGoodMaterial = usedLastGoodMaterial;
    receipt.materialDiagnostic = std::move(materialDiagnostic);
    if (request.parentIntent && !receipt.inspectParent(*request.parentIntent,request.frame,request.structuralRevision,
            request.evaluationRevision,request.projectGeneration,request.helperGeneration,request.renderClipId,owner,
            request.width,request.height,&request.operation,request.proceduralScene,request.sourceCameraNodeId,
            request.runtimeInputs.cameraOverride,&request.runtimeInputs))
        return reject(error,"imported parent completed frame failed expected-request inspection");
    if (parentStaging)
    {
        if (!parentStaging->stage(receipt,error)) return false;
    }
    else output = std::move(receipt);
    published=true;
    error.clear();
    return true;
}

bool ImportedAnimatedScenePayloadExecution::executeComposition(ImportedAnimationEvaluationOwner owner,
    const ImportedAnimatedSceneRequest& request, ImportedAnimatedSceneReceipt& output, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    using scenecomposition::decodeImportedLeafId;
    using scenecomposition::encodeImportedLeafId;
    const auto leafCount = request.compositionLeaves.size();
    const auto leafError = [&](std::size_t leaf, const char* message) {
        error = "Scene Compose refused: imported leaf " + std::to_string(leaf) + " " + message;
        return false;
    };
    if (leafCount < 2 || leafCount > scenecomposition::kMaximumImportedLeaves || !request.proceduralScene)
        return reject(error, "Scene Compose refused: multiple imported leaves require two to eight leaves and their procedural scene");
    if (request.sourceCameraNodeId)
        return reject(error, "Scene Compose refused: a selected source camera with multiple imported leaves is not rendered yet");
    if (!visualimportedobjectlocal::same(request.objectLocalTransform, {}))
        return reject(error, "Imported object local transforms require a native evaluation owner");
    if (request.runtimeInputs.noteInstanceMapping)
        return reject(error, "Scene Compose refused: note instancing of multiple imported leaves is not rendered yet");
    if ((request.material && request.material->vertexModifier) || std::any_of(request.compositionLeaves.begin(),
            request.compositionLeaves.end(), [](const auto& leaf) { return leaf.material && leaf.material->vertexModifier; }))
        return reject(error, "Scene Compose refused: a material vertex processor with multiple imported leaves is not rendered yet");
    if (request.compositionLeaves[0].material)
        return reject(error, "Scene Compose refused: imported leaf 0 material travels as the request material");
    if (!request.importedLights.intensityMultipliers.empty())
        return reject(error, "Scene Compose refused: imported light controls with multiple imported leaves are not rendered yet");
    if (request.evaluationRevision == 0 || request.evaluationRevision < request.structuralRevision)
        return reject(error, "native composition requires the actual nonzero parameter evaluation epoch");
    if (request.structuralRevision == 0 || request.frame.rateNumerator == 0 || request.frame.rateDenominator == 0)
        return reject(error, "imported animated scene frame identity is invalid");
    if (request.width == 0 || request.height == 0 || request.width > 8192 || request.height > 8192
        || static_cast<std::uint64_t>(request.width) * request.height > 67108864u)
        return reject(error, "imported animated scene dimensions exceed the product bound");
    if (!request.projectGeneration || !request.helperGeneration)
        return reject(error, "native composition requires exact project and helper generations");
    if (!compositionBudget_) compositionBudget_ = std::make_shared<NativeCompositionRetentionBudget>();
    if (!sceneAnimationBudget_) sceneAnimationBudget_ = std::make_shared<NativeSceneAnimationCompositionBudget>();

    std::size_t callerLogicalWork = 0;
    constexpr auto callerWorkLimit = 8 * arbitgpu::NativeDeformationCompositionOwner::kMaximumLogicalWork;
    const auto chargeCallerWork = [&](std::size_t count, std::size_t steps) {
        if (count > (callerWorkLimit - callerLogicalWork) / steps)
            return reject(error, "native composition named caller work exceeds capacity");
        callerLogicalWork += count * steps; return true;
    };

    // Selectors arrive in the leaf ID scheme. Each leaf's consumer sees its
    // raw node; the runtime keeps the composed spelling, which the owner decodes.
    // Per-call staging, outside the ledgers and freed on return. What the
    // leaves or the owner keep is copied out and charged where it is kept.
    std::vector<CompositionLeafInput> inputs(leafCount);
    std::vector<visualanimationimport::Request> operations(leafCount);
    std::uint64_t selectedNode = 0;
    std::size_t identity = leafCount;
    for (std::size_t leaf = 0; leaf < leafCount; ++leaf)
    {
        const auto& authored = request.compositionLeaves[leaf];
        auto operation = authored.operation;
        if (!authored.animated && operation.pose.nodeStableId != 0)
            return leafError(leaf, "poses a node without animation");
        const std::uint64_t selector = authored.animated ? operation.pose.nodeStableId : authored.objectNodeStableId;
        if (selector != 0)
        {
            const auto decoded = selector < 0x80000000u
                ? decodeImportedLeafId(static_cast<std::uint32_t>(selector), leafCount) : std::nullopt;
            if (!decoded || decoded->leaf != leaf)
                return leafError(leaf, "selects a node outside its own leaf");
            if (selectedNode != 0)
                return reject(error, "Scene Compose refused: more than one imported leaf selects an object");
            selectedNode = selector;
            if (authored.animated) operation.pose.nodeStableId = decoded->raw;
        }
        if (authored.animated && identity == leafCount) identity = leaf;
        const auto payload = owner == ImportedAnimationEvaluationOwner::Preview
            ? store_.resolvePreview(operation.asset) : store_.resolveExport(operation.asset);
        if (!payload) return leafError(leaf, "exact payload is not resident");
        inputs[leaf].payload = payload;
        inputs[leaf].animated = authored.animated;
        inputs[leaf].operation = operation;
        operations[leaf] = std::move(operation);
    }
    if (identity == leafCount)
        return reject(error, "Scene Compose refused: multiple imported leaves without animation are not rendered yet");

    // A refused frame leaves the epoch where it was, as the one-leaf restore does.
    bool published = false;
    const auto restoreEpoch = [&, payload = compositionEpochPayload_, epoch = compositionEpochIdentity_,
            revision = highestCompositionEvaluationRevision_] {
        if (published) return;
        compositionEpochPayload_ = payload; compositionEpochIdentity_ = epoch;
        highestCompositionEvaluationRevision_ = revision;
    };
    struct RestoreEpoch final { const decltype(restoreEpoch)& action; ~RestoreEpoch() { action(); } } restoreEpochOnExit {restoreEpoch};
    {
        const auto& head = operations[identity];
        const std::array<std::uint64_t,5> epochIdentity {head.sourceStableId, head.deformationStableId,
            request.structuralRevision, request.projectGeneration, request.helperGeneration};
        if (compositionEpochPayload_.lock() == inputs[identity].payload && compositionEpochIdentity_ == epochIdentity)
        {
            if (request.evaluationRevision < highestCompositionEvaluationRevision_)
                return reject(error, "native composition parameter evaluation epoch is stale");
        }
        else highestCompositionEvaluationRevision_ = 0;
        compositionEpochPayload_ = inputs[identity].payload; compositionEpochIdentity_ = epochIdentity;
        highestCompositionEvaluationRevision_ = request.evaluationRevision;
    }

    // Leaf admission depends on exact bytes, selection and clips, not on
    // playback controls or poses, which every frame evaluates afresh.
    const auto sameLeaf = [](const CompositionLeaf& admitted, const CompositionLeafInput& input) {
        const auto& a = admitted.operation; const auto& b = input.operation;
        return admitted.payload == input.payload && admitted.animated == input.animated
            && visualanimationimport::sameAsset(a.asset, b.asset)
            && a.sceneIndex == b.sceneIndex && a.meshStableId == b.meshStableId
            && a.sourceStableId == b.sourceStableId && a.deformationStableId == b.deformationStableId
            && a.animationClipStableId == b.animationClipStableId && a.crossfadeClipStableId == b.crossfadeClipStableId
            && a.clipName == b.clipName && a.crossfadeClipName == b.crossfadeClipName;
    };
    bool admitted = compositionLeaves_.size() == leafCount;
    for (std::size_t leaf = 0; admitted && leaf < leafCount; ++leaf)
        admitted = sameLeaf(compositionLeaves_[leaf], inputs[leaf]);
    // Admission replaces the previous leaves and any single-leaf state. A
    // refusal after it keeps the new leaves, which the next frame reuses.
    // The previous leaves hold their reservations until the new ones admit.
    if (!admitted && !admitCompositionLeaves(inputs, error)) return false;
    for (std::size_t leaf = 0; leaf < leafCount; ++leaf)
    {
        const auto& selector = request.compositionLeaves[leaf];
        const std::uint64_t raw = selector.animated ? operations[leaf].pose.nodeStableId
            : selector.objectNodeStableId == 0 ? 0u
            : decodeImportedLeafId(selector.objectNodeStableId, leafCount)->raw;
        const auto& scene = *compositionLeaves_[leaf].scene;
        if (raw != 0 && std::none_of(scene.objects.begin(), scene.objects.begin() + scene.objectCount,
                [&](const auto& object) { return (object.id.value - 1u) / 65536u + 1u == raw; }))
            return leafError(leaf, "selected object is not drawn by its imported scene selection");
    }

    if (!scene_ || proceduralScene_ != request.proceduralScene)
    {
        // The joined imported scene, each remapped leaf copy, the adapter's
        // leaf copy and the composed result, with their texel storage.
        std::size_t texels = request.proceduralScene->textureTexels.capacity();
        for (const auto& leaf : compositionLeaves_) texels += leaf.scene->textureTexels.capacity();
        if (texels > Visual3DScene::kMaxTextureTexels * (scenecomposition::kMaximumImportedLeaves + 1))
            return reject(error, "Scene Compose refused: imported leaf textures exceed the composed capacity");
        const auto bytes = (leafCount + 3) * sizeof(Visual3DScene) + 4 * texels * sizeof(SceneTexelRgba8)
            + sizeof(NativeCompositionRetentionBudget::Charge) + 8 * sizeof(std::shared_ptr<const void>);
        if (!sceneAnimationBudget_->work(bytes, 2, error)) return false;
        auto charge = NativeCompositionRetentionBudget::reserve(compositionBudget_, bytes, error);
        if (!charge) return false;
        std::vector<std::shared_ptr<const Visual3DScene>> scenes;
        scenes.reserve(leafCount);
        for (const auto& leaf : compositionLeaves_) scenes.push_back(leaf.scene);
        auto composed = composeImportedLeafScenes(*request.proceduralScene, scenes, error);
        if (!composed) return false;
        clearSurfaceMaterial();
        scene_ = retainCompositionAllocation(std::move(composed), charge);
        proceduralScene_ = request.proceduralScene;
        source_.reset();
        motion_ = {}; motionOwner_.reset();
    }

    // Each animated leaf samples its own clip at its own clip time.
    std::vector<ImportedAnimationDeformationEvaluation> evaluations(leafCount);
    std::optional<float> morphWeight;
    for (std::size_t leaf = 0; leaf < leafCount; ++leaf)
    {
        if (!compositionLeaves_[leaf].animated) continue;
        auto& evaluation = evaluations[leaf];
        if (!evaluateCompositionLeaf(owner, compositionLeaves_[leaf], operations[leaf], request.frame,
                request.structuralRevision, evaluation, error)) return false;
        if (compositionLeaves_[leaf].bindings.empty() || !evaluation.deformation)
            return leafError(leaf, "animates only cameras or lights, which multiple imported leaves do not render yet");
        if (!(evaluation.deformation->time() == request.frame))
            return reject(error, "imported animated snapshot presentation frame disagrees with the draw");
        if (!arbitgpu::sceneMotionSnapshotFits(evaluation.deformation))
            return reject(error, "imported animated Motion predecessor exceeds the retained snapshot bound");
        const auto& operation = operations[leaf];
        const float weight = operation.pose.morphEnabled
                || operation.combinationMode == visualanimation::CombinationMode::WeightedBlend
            ? 1.0f : static_cast<float>(operation.playback.weight);
        if (morphWeight && *morphWeight != weight)
            return reject(error, "Scene Compose refused: imported leaves with different animation weights are not rendered yet");
        morphWeight = weight;
    }

    const bool usesMotion = arbitgpu::sceneUsesMotionPass(request.runtimeInputs);
    const auto motionIndex = owner == ImportedAnimationEvaluationOwner::Preview ? 0 : 1;
    const auto rejectMotion = [&](const char* message) {
        motion_[motionIndex] = {};
        return reject(error, message);
    };
    if (usesMotion)
    {
        // Each leaf keeps its skin or morph history through the shared owner.
        // Object and ancestor animation needs per-leaf world tables, which a
        // multi-leaf owner does not retain yet.
        if (!renderer_.supportsRetainedDeformationMotion())
            return rejectMotion("Motion pass requires the native retained deformation consumer");
        for (std::size_t leaf = 0; leaf < leafCount; ++leaf)
        {
            if (!compositionLeaves_[leaf].animated) continue;
            const auto& admittedLeaf = compositionLeaves_[leaf];
            const auto document = admittedLeaf.consumer.admittedDocument();
            const auto& evaluation = evaluations[leaf];
            const auto clip = std::find_if(document->clips.begin(), document->clips.end(),
                [&](const auto& value) { return value.clip && value.clip->id() == evaluation.deformation->clip(); });
            if (clip == document->clips.end())
                return rejectMotion("Motion pass has no exact admitted animation clip");
            if (clip->jointBindings.empty() && clip->morphBindings.empty())
                return rejectMotion("Motion pass requires source-bound skin or morph animation");
            const auto& nodes = admittedLeaf.document->nodes;
            const auto& scene = *admittedLeaf.scene;
            std::vector<bool> objectAncestors(nodes.size(), false);
            for (std::size_t index = 0; index < scene.objectCount; ++index)
            {
                std::optional<std::size_t> node = (scene.objects[index].id.value - 1u) / 65536u;
                for (std::size_t visits = 0; node; ++visits)
                {
                    if (*node >= nodes.size() || visits >= nodes.size())
                        return rejectMotion("Motion pass has invalid imported object ancestry");
                    objectAncestors[*node] = true;
                    node = nodes[*node].parent;
                }
            }
            const auto animatesObject = [&](std::uint64_t target) {
                return target > 0 && target <= objectAncestors.size() && objectAncestors[target - 1u];
            };
            bool objectMotion = false;
            for (const auto& track : clip->clip->tracks())
                if (track.channel() != visualanimation::Channel::MorphWeights && animatesObject(track.target().value))
                    objectMotion = true;
            for (const auto& transform : evaluation.deformation->jointTransforms())
                if (animatesObject(transform.animationTarget().value)) objectMotion = true;
            if (objectMotion)
            {
                motion_[motionIndex] = {};
                return leafError(leaf, "moves an object node, and Motion for multiple imported leaves carries only skin and morph motion yet");
            }
        }
    }

    // Material: leaf 0 keeps its raw object IDs in the composed scene, so a
    // lone leaf 0 binding publishes exactly as for one leaf. Leaf 1+ bindings
    // arrive lowered in their leaf range, object and MaterialId literal, with
    // the digest of that program, so each joins the object collection as its
    // own Render override beside the procedural programs.
    std::vector<std::pair<std::size_t, const surfacematerialbinding::ImportedSceneMaterialRequest*>> leafMaterials;
    if (request.material) leafMaterials.emplace_back(0, &*request.material);
    for (std::size_t leaf = 1; leaf < leafCount; ++leaf)
        if (request.compositionLeaves[leaf].material) leafMaterials.emplace_back(leaf, &*request.compositionLeaves[leaf].material);
    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;
    if (request.material && request.surfacePrograms.empty() && leafMaterials.size() == 1)
    {
        auto material = *request.material;
        if (material.sceneSnapshot != nullptr)
            return reject(error, "imported animated material transport must not carry a foreign scene owner");
        material.scene = scene_->id;
        material.sceneRevision = request.structuralRevision;
        material.sceneSnapshot = scene_;
        if (!publishSurfaceMaterial(material, error))
        {
            if (surfacematerialbinding::hasGraphFrameInput(material.binding)) return false;
            if (owner != ImportedAnimationEvaluationOwner::Preview || lastGoodMaterial_ == nullptr) return false;
            usedLastGoodMaterial = true;
            materialDiagnostic = error;
        }
    }
    else if (!request.surfacePrograms.empty() || !leafMaterials.empty())
    {
        auto programs = request.surfacePrograms;
        for (auto& object : programs)
        {
            const auto mapped = scenecomposition::composedProceduralObjectId(object.program.material.binding.object);
            if (!mapped) return reject(error, "Scene Compose procedural Surface target has no object identity");
            object.program.material.binding.object = *mapped;
            object.program.material.sceneSnapshot = scene_;
        }
        for (const auto& [leaf, material] : leafMaterials)
        {
            if (material->sceneSnapshot != nullptr)
                return reject(error, "imported animated material transport must not carry a foreign scene owner");
            const auto object = material->binding.object;
            if (!scenecomposition::importedLeafOwnsObjectId(object.value, leaf, leafCount))
                return leafError(leaf, "Render Material names an object outside its own leaf");
            const auto* target = visual3d_detail::findById(scene_->objects, scene_->objectCount, object);
            if (!target || target->indexCount == 0)
                return leafError(leaf, "draws no object at its Render Material node and primitive");
            if (target->material.value != geometrysurfacematerial::materialId(*material))
                return leafError(leaf, "Render Material ID does not match its object's material");
            std::string overrideError;
            if (!geometrysurfacematerial::appendImportedOverride(programs, *material,
                    scene_, request.structuralRevision, request.evaluationRevision, overrideError))
                return reject(error, overrideError.c_str());
        }
        videowire::geometry::RuntimeFieldEvaluation fields;
        fields.timelineSeconds = static_cast<double>(request.frame.frame) * request.frame.rateDenominator / request.frame.rateNumerator;
        fields.scoreAt = [&](const auto& operation, const auto& positions, auto& result, auto& diagnostic) {
            return scorefield::evaluateSampleField(operation, positions, request.runtimeInputs.canonicalBlockCFrame, result, diagnostic);
        };
        const auto name = renderer_.backendInfo().backend;
        const auto target = name == "metal" ? videohelper::materialprogram::BackendTarget::Metal
            : name == "opengl" ? videohelper::materialprogram::BackendTarget::OpenGl : videohelper::materialprogram::BackendTarget::Invalid;
        auto admittedMaterial = videorender::fixture3d::admitSurfaceMaterialCollection(scene_, programs, fields, target, error);
        if (!admittedMaterial) return false;
        materialScene_ = scene_;
        lastGoodMaterial_ = std::move(admittedMaterial);
        lastGoodMaterialRevision_ = request.structuralRevision;
        rejectedMaterialRevision_ = 0;
    }
    else clearSurfaceMaterial();
    if (owner == ImportedAnimationEvaluationOwner::Export && rejectedMaterialRevision_ > lastGoodMaterialRevision_)
        return reject(error, "current animated surface material revision is not exportable");

    // One owner holds every leaf. It is rebuilt when any leaf's clip, the
    // revisions or the generations change.
    const auto& headOperation = operations[identity];
    const auto clipOf = [&](std::size_t leaf) { return evaluations[leaf].deformation->clip(); };
    bool current = source_ && source_->compositionOwner
        && source_->sourceStableId == headOperation.sourceStableId
        && source_->deformationStableId == headOperation.deformationStableId
        && source_->structuralRevision == request.structuralRevision
        && source_->evaluationRevision == request.evaluationRevision
        && source_->clip == clipOf(identity)
        && source_->compositionOwner->projectGeneration() == request.projectGeneration
        && source_->compositionOwner->helperGeneration() == request.helperGeneration
        && source_->compositionOwner->leaves_.size() == leafCount;
    for (std::size_t leaf = 0; current && leaf < leafCount; ++leaf)
    {
        const auto& input = source_->compositionOwner->leaves_[leaf].input;
        current = input.payload.get() == compositionLeaves_[leaf].payload.get()
            && static_cast<bool>(input.animation) == compositionLeaves_[leaf].animated
            && (!input.imported || input.imported->clip == clipOf(leaf));
    }
    if (!current)
    {
        const auto n = scene_->objectCount;
        std::size_t draws = 0;
        for (const auto& leaf : compositionLeaves_) draws += leaf.bindings.size();
        if (n == 0 || n > 64 || draws > 64)
            return reject(error, "native composition caller cardinalities exceed private owner admission");
        // Leaf draw assembly and final typed wrappers each copy actual
        // joint/morph vectors. The private owner's own debit is separate.
        std::size_t wrapperBytes = (1 + leafCount) * sizeof(arbitgpu::NativeDeformationScene)
            + (28 + 8 * leafCount) * sizeof(std::shared_ptr<const void>) + sizeof(NativeCompositionRetentionBudget::Charge)
            + leafCount * sizeof(arbitgpu::NativeDeformationCompositionLeaf)
            + n * (sizeof(arbitgpu::NativeDeformationScene) + sizeof(std::shared_ptr<const arbitgpu::NativeDeformationScene>)
                + sizeof(arbitgpu::NativeDeformationCompositionMapping))
            + draws * (sizeof(arbitgpu::NativeDeformationScene) + sizeof(std::shared_ptr<const arbitgpu::NativeDeformationScene>));
        for (const auto& leaf : compositionLeaves_)
            for (const auto& binding : leaf.bindings)
            {
                if (binding.jointBaseTransforms.size() > 64 || binding.morphBaseWeights.size() > 8)
                    return reject(error, "native composition caller binding cardinalities exceed private admission");
                wrapperBytes += 3 * (binding.jointBaseTransforms.size() * sizeof(arbitgpu::NativeDeformationJointBaseTransform)
                    + binding.morphBaseWeights.size() * sizeof(float));
            }
        wrapperBytes += proceduralScene_->objectCount * (sizeof(arbitgpu::NativeDeformationJointBaseTransform)
            + 4 * sizeof(std::shared_ptr<const void>));
        if (wrapperBytes > NativeCompositionRetentionBudget::kCallerMaximumBytes)
            return reject(error, "native composition wrapper storage exceeds its source-derived cap");
        const auto ownerBytes = 2 * arbitgpu::NativeDeformationCompositionOwner::singleLeafBytes()
            + 4096 * sizeof(bool) + 12288 * sizeof(bool)
            + n * (sizeof(arbitgpu::NativeDeformationScene) + sizeof(visualdeformation::DeformationAsset)
                + sizeof(visualdeformation::Skin) + sizeof(visualdeformation::Joint) + sizeof(visualdeformation::Mesh)
                + sizeof(visualdeformation::JointWeightSet) + sizeof(arbitgpu::NativeDeformationJointBaseTransform)
                + sizeof(std::vector<std::uint32_t>) + sizeof(std::vector<float>))
            + 2 * 4096 * 4 * (sizeof(std::uint32_t) + sizeof(float)) + 2 * n * sizeof(arbitgpu::NativeDeformationFrameData)
            + arbitgpu::NativeDeformationCompositionOwner::leafListBytes(leafCount);
        // Leaf originals stay charged by each leaf's own reservation, which
        // the owner's aliases keep. Only the procedural original is new here.
        const auto inputBytes = sizeof(Visual3DScene)
            + request.proceduralScene->textureTexels.capacity() * sizeof(SceneTexelRgba8);
        if (!chargeCallerWork(inputBytes, 2) || !chargeCallerWork(wrapperBytes, 2) || !chargeCallerWork(n * n, 16))
            return false;
        if (inputBytes > NativeCompositionRetentionBudget::kMaximumBytes - wrapperBytes
            || ownerBytes > NativeCompositionRetentionBudget::kMaximumBytes - wrapperBytes - inputBytes)
            return reject(error, "native composition full owner inputs exceed logical CPU capacity");
        const auto ownerCharge = NativeCompositionRetentionBudget::reserve(compositionBudget_,
            wrapperBytes + ownerBytes + inputBytes, error);
        if (!ownerCharge) return false;

        std::vector<arbitgpu::NativeDeformationCompositionLeaf> ownerLeaves(leafCount);
        std::vector<arbitgpu::NativeDeformationCompositionMapping> mappings;
        mappings.reserve(n);
        for (std::size_t leaf = 0; leaf < leafCount; ++leaf)
        {
            const auto& admittedLeaf = compositionLeaves_[leaf];
            // The owner aliases the leaf's document, scene and animation
            // headers (charges[0]) as well as their decoded closure
            // (charges[1]), so each alias keeps both reservations alive.
            const auto retention = [&admittedLeaf](auto value) {
                return retainSceneAnimationAllocation(
                    retainSceneAnimationAllocation(std::move(value), admittedLeaf.charges[0]), admittedLeaf.charges[1]);
            };
            auto& input = ownerLeaves[leaf];
            input.payload = retention(admittedLeaf.payload);
            input.document = retention(admittedLeaf.document);
            input.selectedScene = admittedLeaf.operation.sceneIndex;
            input.selectedMesh = admittedLeaf.operation.meshStableId;
            input.scene = retention(admittedLeaf.scene);
            if (admittedLeaf.animated)
            {
                const auto animation = admittedLeaf.consumer.admittedDocument();
                if (!animation || !animation->deformation) return leafError(leaf, "has no admitted deformation");
                input.animation = retention(animation);
                const auto& operation = operations[leaf];
                auto source = std::make_shared<arbitgpu::NativeDeformationScene>();
                source->sourceStableId = operation.sourceStableId;
                source->deformationStableId = operation.deformationStableId;
                source->structuralRevision = request.structuralRevision;
                source->evaluationRevision = request.evaluationRevision;
                source->clip = clipOf(leaf);
                source->scene = input.scene;
                source->logicalCompositionRetention = ownerCharge;
                source->deformation = retainCompositionAllocation(animation->deformation, ownerCharge);
                source->draws.reserve(admittedLeaf.bindings.size());
                for (const auto& binding : admittedLeaf.bindings)
                {
                    auto draw = std::make_shared<arbitgpu::NativeDeformationScene>();
                    draw->sourceStableId = source->sourceStableId; draw->deformationStableId = source->deformationStableId;
                    draw->structuralRevision = source->structuralRevision; draw->evaluationRevision = source->evaluationRevision;
                    draw->clip = source->clip; draw->scene = source->scene; draw->deformation = source->deformation;
                    draw->logicalCompositionRetention = ownerCharge;
                    draw->jointBaseTransforms.reserve(binding.jointBaseTransforms.size());
                    draw->mesh = binding.mesh;
                    draw->skin = binding.skin;
                    draw->animationNodeStableId = binding.nodeIndex + 1u;
                    draw->object = {static_cast<std::uint32_t>(binding.nodeIndex * 65536u + 1u)};
                    draw->morphBaseWeights = binding.morphBaseWeights;
                    draw->batchMember = true;
                    for (const auto& base : binding.jointBaseTransforms)
                        draw->jointBaseTransforms.push_back({base.skin, base.joint,
                            base.translation, base.rotation, base.scale, base.matrix});
                    source->draws.push_back(std::move(draw));
                }
                if (!source->draws.empty())
                {
                    source->mesh = source->draws.front()->mesh;
                    source->object = source->draws.front()->object;
                }
                input.imported = retainCompositionAllocation<arbitgpu::NativeDeformationScene>(std::move(source), ownerCharge);
            }
            const auto& scene = *admittedLeaf.scene;
            for (std::size_t i = 0; i < scene.objectCount; ++i)
            {
                const auto raw = scene.objects[i].id;
                const auto composed = composedLeafObjectId(leaf, leafCount, raw);
                const auto node = raw.isValid() ? encodeImportedLeafId(leaf, leafCount, (raw.value - 1u) / 65536u + 1u)
                                                : std::nullopt;
                if (!composed || !node)
                    return leafError(leaf, "object has no exact composed identity");
                mappings.push_back({arbitgpu::NativeDeformationCompositionMapping::Kind::Imported, raw, *composed, *node});
            }
        }
        for (std::size_t i = 0; i < proceduralScene_->objectCount; ++i)
        {
            const auto id = proceduralScene_->objects[i].id;
            const auto target = scenecomposition::composedProceduralObjectId(id);
            if (!target) return reject(error, "native composition procedural mapping has no exact composed identity");
            mappings.push_back({arbitgpu::NativeDeformationCompositionMapping::Kind::Procedural, id, *target, 0});
        }
        auto compositionOwner = arbitgpu::NativeDeformationCompositionOwner::admit(std::move(ownerLeaves),
            retainCompositionAllocation(proceduralScene_, ownerCharge), retainCompositionAllocation(scene_, ownerCharge),
            mappings, request.projectGeneration, request.helperGeneration, error, request.evaluationRevision);
        if (!compositionOwner) return false;
        const auto& head = *compositionOwner->imported_;
        auto batch = std::make_shared<arbitgpu::NativeDeformationScene>();
        batch->sourceStableId = head.sourceStableId; batch->deformationStableId = head.deformationStableId;
        batch->structuralRevision = head.structuralRevision; batch->evaluationRevision = request.evaluationRevision;
        batch->clip = head.clip; batch->scene = compositionOwner->scene(); batch->deformation = head.deformation;
        batch->compositionOwner = compositionOwner; batch->draws.reserve(compositionOwner->memberCount());
        for (std::size_t i = 0; i < compositionOwner->memberCount(); ++i)
        {
            auto draw = std::make_shared<arbitgpu::NativeDeformationScene>(*compositionOwner->originalMember(i));
            draw->scene = compositionOwner->scene(); draw->object = compositionOwner->memberObject(i);
            draw->compositionOwner = compositionOwner;
            draw->draws.clear(); draw->batchMember = true;
            batch->draws.push_back(std::move(draw));
        }
        if (batch->draws.empty()) return reject(error, "native composition has no complete native members");
        batch->mesh = batch->draws.front()->mesh; batch->object = batch->draws.front()->object;
        source_ = retainCompositionAllocation<arbitgpu::NativeDeformationScene>(std::move(batch), ownerCharge);
        motion_ = {}; motionOwner_.reset();
    }

    // One frame holds one sample per leaf in leaf order; static leaves none.
    std::shared_ptr<const arbitgpu::NativeDeformationCompositionFrame> compositionFrame;
    std::shared_ptr<const NativeCompositionRetentionBudget::Charge> frameCharge;
    {
        std::size_t bytes = 0;
        if (!compositionFrameBytes(*source_->compositionOwner, evaluations[identity], bytes, error)) return false;
        const auto leafList = arbitgpu::NativeDeformationCompositionFrame::leafListBytes(leafCount);
        if (leafList > NativeCompositionRetentionBudget::kMaximumBytes - bytes)
            return reject(error, "native composition caller snapshot storage exceeds logical capacity");
        bytes += leafList;
        for (std::size_t leaf = 0; leaf < leafCount; ++leaf)
        {
            if (!compositionLeaves_[leaf].animated) continue;
            const auto& evaluation = evaluations[leaf];
            if (!chargeCallerWork(evaluation.deformation->morphWeights().size(), 32)
                || (evaluation.sceneAnimation && !chargeCallerWork(evaluation.sceneAnimation->tracks().size(), 32)))
                return false;
            if (leaf != identity && !compositionLeafSampleBytes(evaluation, bytes, error)) return false;
        }
        frameCharge = NativeCompositionRetentionBudget::reserve(compositionBudget_, bytes, error);
        if (!frameCharge) return false;
        std::vector<arbitgpu::NativeDeformationCompositionLeafSample> samples(leafCount);
        for (std::size_t leaf = 0; leaf < leafCount; ++leaf)
        {
            if (!compositionLeaves_[leaf].animated) continue;
            auto& evaluation = evaluations[leaf];
            evaluation.deformation = retainCompositionAllocation(evaluation.deformation, frameCharge);
            if (evaluation.sceneAnimation)
                evaluation.sceneAnimation = retainCompositionAllocation(evaluation.sceneAnimation, frameCharge);
            samples[leaf].deformation = evaluation.deformation;
            samples[leaf].sceneAnimation = evaluation.sceneAnimation;
        }
        compositionFrame = arbitgpu::NativeDeformationCompositionFrame::admit(source_->compositionOwner,
            std::move(samples), request.frame, request.projectGeneration, request.helperGeneration, error);
        if (!compositionFrame) return false;
    }

    videorender::animation3d::RenderedDeformationFrame frame;
    const auto materialScene = materialScene_.lock();
    const auto material = materialScene == scene_ ? lastGoodMaterial_ : nullptr;
    const auto program = material ? material->nativeProgram() : nullptr;
    const bool hasVertexProgram = program && (program->vertexProgram
            || std::any_of(program->objectPrograms.begin(), program->objectPrograms.end(),
                [](const auto& value) { return value && value->vertexProgram; }));
    if (usesMotion && hasVertexProgram)
    {
        const auto refusal = [](const auto& value) {
            return value && value->vertexProgram ? videowire::vertexModifierPriorGeometryRefusal(
                value->vertexProgram->vertexInstructions(), videowire::VertexModifierPriorGeometry::skinnedPose) : nullptr;
        };
        const char* message = refusal(program);
        for (const auto& value : program->objectPrograms)
            if (!message) message = refusal(value);
        if (message) return rejectMotion(message);
    }
    const auto exactSeconds = static_cast<double>(request.frame.frame)
        * static_cast<double>(request.frame.rateDenominator)
        / static_cast<double>(request.frame.rateNumerator);
    if (!std::isfinite(exactSeconds) || std::abs(exactSeconds) > surfacematerial::kMaximumEvaluationMagnitude)
        return reject(error, "imported animated material time is out of bounds");
    auto runtimeInputs = request.runtimeInputs;
    runtimeInputs.compositionFrame = compositionFrame;
    runtimeInputs.canonicalBlockCFrame.reset();
    runtimeInputs.objectNodeStableId = selectedNode;
    runtimeInputs.timeSeconds = static_cast<float>(exactSeconds);
    runtimeInputs.morphWeight = *morphWeight;

    // The motion owner names the identity leaf; the composition frame names
    // the owner of every leaf, so a changed leaf clip starts a new history.
    const arbitgpu::NativeSceneMotionOwner motionOwner {{ source_->sourceStableId,
        source_->deformationStableId, source_->clip.value, source_->structuralRevision,
        request.projectGeneration, request.helperGeneration,
        static_cast<std::uint64_t>(request.renderClipId), runtimeInputs.objectNodeStableId },
        request.evaluationRevision };
    const auto device = renderer_.backendInfo();
    const auto motionDevice = device.backend + "\n" + device.device;
    if ((motionOwner_ && !(*motionOwner_ == motionOwner)) || motionDevice_ != motionDevice)
        motion_ = {};
    motionOwner_ = motionOwner; motionDevice_ = motionDevice;
    auto& motion = motion_[motionIndex];
    runtimeInputs.previousMotion = motion.predecessor(request.frame.frame,
        request.frame.rateNumerator, request.frame.rateDenominator,
        request.width, request.height, motionOwner, runtimeInputs);
    {
        const auto work = compositionFrame->logicalWork();
        const auto previousWork = runtimeInputs.previousMotion && runtimeInputs.previousMotion->compositionFrame
            ? runtimeInputs.previousMotion->compositionFrame->logicalWork() : 0;
        const auto preparationWork = 3 * work + 2 * previousWork + source_->compositionOwner->logicalWork();
        if (preparationWork > callerWorkLimit || callerLogicalWork > callerWorkLimit - preparationWork)
            return reject(error, "native composition caller capture and validation work exceed capacity");
    }
    if (runtimeInputs.previousMotion
        && (!arbitgpu::validSceneMotionImportedNodeWorldMatrices(*runtimeInputs.previousMotion, false)
            || runtimeInputs.previousMotion->importedNodeWorldMatrices))
        return rejectMotion("imported Motion node world table is malformed or exceeds the snapshot bound");

    const auto& identityDeformation = evaluations[identity].deformation;
    const bool rendered = owner == ImportedAnimationEvaluationOwner::Preview
        ? renderer_.renderPreview(source_, identityDeformation, program, runtimeInputs, request.width, request.height,
            videorender::animation3d::kNativeGpuCapability, frame, error)
        : renderer_.renderExport(source_, identityDeformation, program, runtimeInputs, request.width, request.height,
            videorender::animation3d::kNativeGpuCapability, frame, error);
    if (!rendered) return false;
    if (!motion.commit(request.frame.frame, request.frame.rateNumerator, request.frame.rateDenominator,
            request.width, request.height, runtimeInputs, identityDeformation, motionOwner))
        motion = {};

    ImportedAnimatedSceneReceipt receipt;
    receipt.owner = owner;
    receipt.payload = source_->compositionOwner->payload_;
    receipt.source = source_;
    receipt.evaluation = std::move(evaluations[identity]);
    receipt.frame = std::move(frame);
    receipt.runtimeInputs = std::move(runtimeInputs);
    receipt.usedLastGoodMaterial = usedLastGoodMaterial;
    receipt.materialDiagnostic = std::move(materialDiagnostic);
    if (!receipt.valid())
        return reject(error, "native composition frame does not match its issued authority");
    output = std::move(receipt);
    published = true;
    error.clear();
    return true;
}
} // namespace videohelper::modelpayload
