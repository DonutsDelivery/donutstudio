#include "gpu_backend/backend.h"
#include "gltf_glb.h"
#include "../../shared/VisualImportedSceneIdentity.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>

namespace arbitgpu
{
namespace
{
using Matrix = std::array<float, 16>;

Matrix multiply (const Matrix& left, const Matrix& right) noexcept
{
    Matrix result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row] += left[inner * 4 + row]
                    * right[column * 4 + inner];
    return result;
}

Matrix compose (const std::array<float, 3>& translation,
                const std::array<float, 4>& rotation,
                const std::array<float, 3>& scale) noexcept
{
    const auto x = rotation[0], y = rotation[1], z = rotation[2], w = rotation[3];
    const auto xx = x * x, yy = y * y, zz = z * z;
    const auto xy = x * y, xz = x * z, yz = y * z;
    const auto wx = w * x, wy = w * y, wz = w * z;
    return {
        (1.0f - 2.0f * (yy + zz)) * scale[0],
        (2.0f * (xy + wz)) * scale[0],
        (2.0f * (xz - wy)) * scale[0], 0.0f,
        (2.0f * (xy - wz)) * scale[1],
        (1.0f - 2.0f * (xx + zz)) * scale[1],
        (2.0f * (yz + wx)) * scale[1], 0.0f,
        (2.0f * (xz + wy)) * scale[2],
        (2.0f * (yz - wx)) * scale[2],
        (1.0f - 2.0f * (xx + yy)) * scale[2], 0.0f,
        translation[0], translation[1], translation[2], 1.0f
    };
}

Matrix inverseTrs (const HarmonicMIDI::grid::SceneTransform3D& transform) noexcept
{
    const std::array<float, 4> inverseRotation {
        -transform.rotation.x, -transform.rotation.y,
        -transform.rotation.z, transform.rotation.w
    };
    const Matrix inverseScale {
        1.0f / transform.scale.x, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / transform.scale.y, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f / transform.scale.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    const auto rotation = compose ({ 0.0f, 0.0f, 0.0f }, inverseRotation,
                                   { 1.0f, 1.0f, 1.0f });
    const Matrix translation {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        -transform.translation.x, -transform.translation.y,
        -transform.translation.z, 1.0f
    };
    return multiply (inverseScale, multiply (rotation, translation));
}

bool finite (float value) noexcept { return std::isfinite (value); }

template <std::size_t Size>
std::array<float, Size> weightedBlend (const std::array<float, Size>& base,
                                       const std::array<float, Size>& sampled,
                                       float weight) noexcept
{
    std::array<float, Size> result {};
    for (std::size_t index = 0; index < Size; ++index)
        result[index] = base[index] + (sampled[index] - base[index]) * weight;
    return result;
}

std::array<float, 4> weightedRotation (const std::array<float, 4>& base,
                                       std::array<float, 4> sampled,
                                       float weight) noexcept
{
    const float dot = base[0] * sampled[0] + base[1] * sampled[1]
        + base[2] * sampled[2] + base[3] * sampled[3];
    if (dot < 0.0f)
        for (auto& value : sampled) value = -value;
    auto result = weightedBlend(base, sampled, weight);
    const float norm = std::sqrt(result[0] * result[0] + result[1] * result[1]
        + result[2] * result[2] + result[3] * result[3]);
    if (norm > 0.0f)
        for (auto& value : result) value /= norm;
    return result;
}

std::array<float, 4> multiplyRotation (const std::array<float, 4>& left,
                                       const std::array<float, 4>& right) noexcept
{
    return {
        left[3] * right[0] + left[0] * right[3]
            + left[1] * right[2] - left[2] * right[1],
        left[3] * right[1] - left[0] * right[2]
            + left[1] * right[3] + left[2] * right[0],
        left[3] * right[2] + left[0] * right[1]
            - left[1] * right[0] + left[2] * right[3],
        left[3] * right[3] - left[0] * right[0]
            - left[1] * right[1] - left[2] * right[2]
    };
}

template <std::size_t Size>
std::array<float, Size> addRelative (const std::array<float, Size>& base,
                                     const std::array<float, Size>& sampled,
                                     const std::array<float, Size>& identity,
                                     float weight) noexcept
{
    std::array<float, Size> result {};
    for (std::size_t index = 0; index < Size; ++index)
        result[index] = base[index] + (sampled[index] - identity[index]) * weight;
    return result;
}

template <std::size_t Size>
std::array<float, Size> multiplyRelative (const std::array<float, Size>& base,
                                          const std::array<float, Size>& sampled,
                                          const std::array<float, Size>& identity,
                                          float weight) noexcept
{
    std::array<float, Size> result {};
    for (std::size_t index = 0; index < Size; ++index)
        result[index] = base[index]
            * (identity[index] + (sampled[index] - identity[index]) * weight);
    return result;
}

template <std::size_t Size>
bool finiteArray (const std::array<float, Size>& values) noexcept
{
    return std::all_of (values.begin(), values.end(), finite);
}

const HarmonicMIDI::grid::SceneObjectRecord* findObject (
    const HarmonicMIDI::grid::Visual3DScene& scene,
    HarmonicMIDI::grid::SceneObjectId id) noexcept
{
    for (std::size_t index = 0; index < scene.objectCount; ++index)
        if (scene.objects[index].id == id) return &scene.objects[index];
    return nullptr;
}
} // namespace

static bool prepareOriginalNativeDeformationFrame (
    const NativeDeformationScene& source,
    const visualdeformation::AnimationDeformationSnapshot& snapshot,
    NativeDeformationFrameData& output,
    std::string& error,
    NativeSceneMotionSample::ImportedNodeWorldMatrices* sampledWorlds,
    const Matrix* originalInverse = nullptr)
{
    using namespace visualdeformation;
    error.clear();
    if (sampledWorlds) sampledWorlds->clear();
    NativeSceneMotionSample::ImportedNodeWorldMatrices candidateWorlds;
    NativeDeformationFrameData candidate;
    if (!source.draws.empty())
    {
        if (!source.scene || !HarmonicMIDI::grid::validateVisual3DScene(*source.scene).valid()
            || source.draws.size() > HarmonicMIDI::grid::Visual3DScene::kMaxObjects
            || source.batchMember)
        { error = "native deformation scene batch exceeds its bounds"; return false; }
        std::vector<bool> occupied(source.scene->vertexCount, false);
        for (const auto& draw : source.draws)
        {
            if (!draw || !draw->draws.empty() || !draw->batchMember
                || draw->scene != source.scene || draw->deformation != source.deformation
                || draw->retainDeformedGeometry != source.retainDeformedGeometry
                || draw->sourceStableId != source.sourceStableId
                || draw->deformationStableId != source.deformationStableId
                || draw->structuralRevision != source.structuralRevision || draw->clip != source.clip
                || draw->compositionOwner != source.compositionOwner)
            { error = "native deformation scene batch has a foreign draw owner"; return false; }
            NativeDeformationFrameData value;
            NativeSceneMotionSample::ImportedNodeWorldMatrices drawWorlds;
            if (!prepareOriginalNativeDeformationFrame(*draw, snapshot, value, error,
                    sampledWorlds ? &drawWorlds : nullptr)) return false;
            for (const auto& sampled : drawWorlds)
            {
                const auto found = std::find_if(candidateWorlds.begin(), candidateWorlds.end(),
                    [&](const auto& entry) { return entry.nodeStableId == sampled.nodeStableId; });
                if (found == candidateWorlds.end()) candidateWorlds.push_back(sampled);
                else if (found->world != sampled.world)
                { error = "native deformation batch has conflicting sampled node worlds"; return false; }
            }
            const auto* object = findObject(*source.scene, draw->object);
            for (std::size_t index = object->firstVertex; index < object->firstVertex + value.vertexCount; ++index)
            {
                if (occupied[index])
                { error = "native deformation scene batch overlaps mesh topology"; return false; }
                occupied[index] = true;
            }
        }
        output = candidate;
        if (sampledWorlds) *sampledWorlds = std::move(candidateWorlds);
        return true;
    }
    if ((snapshot.combinationMode() != visualanimation::CombinationMode::Replace
         && snapshot.combinationMode() != visualanimation::CombinationMode::WeightedBlend
         && snapshot.combinationMode() != visualanimation::CombinationMode::Add
         && snapshot.combinationMode() != visualanimation::CombinationMode::Multiply
         && snapshot.combinationMode()
             != visualanimation::CombinationMode::AddAfterImportedAnimation)
        || ! std::isfinite(snapshot.combinationWeight())
        || snapshot.combinationWeight() < 0.0 || snapshot.combinationWeight() > 1.0)
    {
        error = "native deformation combination mode or weight is invalid";
        return false;
    }
    const auto combinationMode = snapshot.combinationMode();
    const auto combinationWeight = static_cast<float>(snapshot.combinationWeight());
    if (source.sourceStableId == 0 || source.deformationStableId == 0
        || source.structuralRevision == 0
        || ! source.clip.isValid() || ! source.mesh.isValid() || ! source.object.isValid()
        || ! source.scene || ! source.deformation)
    {
        error = "native deformation source identities are incomplete";
        return false;
    }
    if (snapshot.clip() != source.clip
        || snapshot.revision() != source.structuralRevision
        || snapshot.time().rateNumerator == 0 || snapshot.time().rateDenominator == 0
        || ! std::isfinite (snapshot.requestedTimeSeconds())
        || ! std::isfinite (snapshot.sampleTimeSeconds()))
    {
        error = "native deformation snapshot identity is incompatible";
        return false;
    }
    const auto sceneValidation = HarmonicMIDI::grid::validateVisual3DScene (*source.scene);
    if (! sceneValidation.valid())
    {
        error = "native deformation scene validation failed";
        return false;
    }
    if (!source.batchMember && (source.scene->objectCount != 1 || source.scene->materialCount != 1
        || source.scene->lightCount > 1 || source.scene->cameraCount != 1))
    {
        error = "native deformation admits one object, one material, one camera, and at most one light";
        return false;
    }
    const auto* object = findObject (*source.scene, source.object);
    const auto* mesh = source.deformation->findMesh (source.mesh);
    if (object == nullptr || mesh == nullptr || (object->parent.isValid() && !originalInverse)
        || (!source.batchMember && (object != &source.scene->objects[0]
            || object->firstVertex != 0 || object->vertexCount != source.scene->vertexCount
            || mesh->vertexCount() != object->vertexCount))
        || object->firstVertex > source.scene->vertexCount
        || mesh->vertexCount() > source.scene->vertexCount - object->firstVertex
        || mesh->vertexCount() == 0 || mesh->vertexCount() > kNativeDeformationMaxVertices)
    {
        error = "native deformation object and mesh ranges are incompatible";
        return false;
    }
    if (!source.batchMember && (source.scene->materials[0].opacity != 1.0f
        || (source.scene->lightCount == 1
            && source.scene->lights[0].kind
                != HarmonicMIDI::grid::SceneLightKind::Directional)))
    {
        error = "native deformation scene is outside the opaque fixture subset: "
            "opaque material, and zero lights or one Directional light";
        return false;
    }
    if (mesh->jointWeightSets().size() > 1
        || mesh->morphTargets().size() > kNativeDeformationMaxMorphTargets)
    {
        error = "native deformation mesh exceeds the bounded influence or morph subset";
        return false;
    }
    for (const auto& target : mesh->morphTargets())
        if (target.hasTangentDeltas())
        {
            error = "native deformation does not admit tangent morph deltas";
            return false;
        }
    candidate.vertexCount = static_cast<std::uint32_t> (mesh->vertexCount());
    candidate.morphTargetCount = static_cast<std::uint32_t> (mesh->morphTargets().size());
    const auto skinId = source.skin.value_or(mesh->skin());
    auto pose = snapshot.pose();
    if (source.batchMember && (pose.meshStableId != mesh->id().value
        || (pose.nodeStableId != 0 && pose.nodeStableId != source.animationNodeStableId))) pose = {};
    if (!visualanimation::validPoseControls(pose)
        || ((pose.boneEnabled || pose.morphEnabled) && pose.meshStableId != mesh->id().value)
        || (pose.morphEnabled && pose.morphTargetIndex >= mesh->morphTargets().size())
        || (pose.boneEnabled && !skinId.isValid()))
    {
        error = "native deformation selected bone or morph target is incompatible";
        return false;
    }
    if (! source.morphBaseWeights.empty()
        && source.morphBaseWeights.size() != mesh->morphTargets().size())
    {
        error = "native deformation base morph weights do not match exact target identity";
        return false;
    }
    for (std::size_t index = 0; index < source.morphBaseWeights.size(); ++index)
    {
        if (! finite (source.morphBaseWeights[index])
            || std::abs (source.morphBaseWeights[index]) > 16.0f)
        {
            error = "native deformation base morph weight is nonfinite or out of range";
            return false;
        }
        candidate.morphWeights[index] = source.morphBaseWeights[index];
    }

    const Skin* skin = nullptr;
    if (skinId.isValid())
    {
        skin = source.deformation->findSkin (skinId);
        if (skin == nullptr || skin->joints().empty()
            || skin->joints().size() > kNativeDeformationMaxJoints
            || mesh->jointWeightSets().size() != 1
            || source.jointBaseTransforms.size() != skin->joints().size())
        {
            error = "native deformation skin exceeds or does not satisfy the bounded subset";
            return false;
        }
        candidate.jointCount = static_cast<std::uint32_t> (skin->joints().size());
    }
    else if (! source.jointBaseTransforms.empty() || ! mesh->jointWeightSets().empty())
    {
        error = "native deformation declares joint data without an exact skin";
        return false;
    }

    if (skin != nullptr)
    {
        std::unordered_map<std::uint64_t, std::size_t> indices;
        indices.reserve (skin->joints().size());
        std::vector<std::array<float, 3>> translations;
        std::vector<std::array<float, 4>> rotations;
        std::vector<std::array<float, 3>> scales;
        translations.reserve (skin->joints().size());
        rotations.reserve (skin->joints().size());
        scales.reserve (skin->joints().size());
        for (std::size_t index = 0; index < skin->joints().size(); ++index)
        {
            const auto& joint = skin->joints()[index];
            const auto& base = source.jointBaseTransforms[index];
            if (base.skin != skin->id() || base.joint != joint.id()
                || ! finiteArray (base.translation) || ! finiteArray (base.rotation)
                || ! finiteArray (base.scale)
                || (base.matrix && !finiteArray(*base.matrix))
                || base.scale[0] == 0.0f || base.scale[1] == 0.0f || base.scale[2] == 0.0f)
            {
                error = "native deformation base pose does not preserve exact joint identity";
                return false;
            }
            const float rotationNorm = base.rotation[0] * base.rotation[0]
                + base.rotation[1] * base.rotation[1]
                + base.rotation[2] * base.rotation[2]
                + base.rotation[3] * base.rotation[3];
            if (! finite (rotationNorm) || std::abs (rotationNorm - 1.0f) > 0.001f)
            {
                error = "native deformation base joint rotation is not normalized";
                return false;
            }
            indices.emplace (joint.id().value, index);
            translations.push_back (base.translation);
            rotations.push_back (base.rotation);
            scales.push_back (base.scale);
        }
        std::vector<bool> animated (skin->joints().size(), false);
        for (const auto& evaluation : snapshot.jointTransforms())
        {
            if (evaluation.skin() != skin->id())
            {
                if (source.batchMember) continue;
                error = "native deformation snapshot contains a different skin identity";
                return false;
            }
            const auto found = indices.find (evaluation.joint().value);
            if (found == indices.end() || animated[found->second])
            {
                error = "native deformation snapshot joint identity is missing or duplicated";
                return false;
            }
            animated[found->second] = true;
            if (evaluation.hasTranslation())
            {
                const std::array<float, 3> identity {};
                if (combinationMode == visualanimation::CombinationMode::WeightedBlend)
                    translations[found->second] = weightedBlend(
                        translations[found->second], evaluation.translation(),
                        combinationWeight);
                else if (combinationMode == visualanimation::CombinationMode::Multiply)
                    translations[found->second] = multiplyRelative(
                        translations[found->second], evaluation.translation(), identity,
                        combinationWeight);
                else if (combinationMode == visualanimation::CombinationMode::Add
                         || combinationMode
                             == visualanimation::CombinationMode::AddAfterImportedAnimation)
                    translations[found->second] = addRelative(
                        translations[found->second], evaluation.translation(), identity,
                        combinationWeight);
                else translations[found->second] = evaluation.translation();
            }
            if (evaluation.hasRotation())
            {
                const std::array<float, 4> identity { 0.0f, 0.0f, 0.0f, 1.0f };
                if (combinationMode == visualanimation::CombinationMode::WeightedBlend)
                    rotations[found->second] = weightedRotation(
                        rotations[found->second], evaluation.rotation(), combinationWeight);
                else if (combinationMode == visualanimation::CombinationMode::Add
                         || combinationMode == visualanimation::CombinationMode::Multiply
                         || combinationMode
                             == visualanimation::CombinationMode::AddAfterImportedAnimation)
                {
                    const auto delta = weightedRotation(identity, evaluation.rotation(),
                                                        combinationWeight);
                    rotations[found->second] = combinationMode
                            == visualanimation::CombinationMode::Add
                        ? multiplyRotation(delta, rotations[found->second])
                        : multiplyRotation(rotations[found->second], delta);
                }
                else rotations[found->second] = evaluation.rotation();
            }
            if (evaluation.hasScale())
            {
                const std::array<float, 3> identity { 1.0f, 1.0f, 1.0f };
                if (combinationMode == visualanimation::CombinationMode::WeightedBlend)
                    scales[found->second] = weightedBlend(
                        scales[found->second], evaluation.scale(), combinationWeight);
                else if (combinationMode == visualanimation::CombinationMode::Multiply)
                    scales[found->second] = multiplyRelative(
                        scales[found->second], evaluation.scale(), identity,
                        combinationWeight);
                else if (combinationMode == visualanimation::CombinationMode::Add
                         || combinationMode
                             == visualanimation::CombinationMode::AddAfterImportedAnimation)
                    scales[found->second] = addRelative(
                        scales[found->second], evaluation.scale(), identity,
                        combinationWeight);
                else scales[found->second] = evaluation.scale();
            }
        }
        if (pose.boneEnabled)
        {
            const auto found = indices.find(pose.boneStableId);
            if (found == indices.end())
            {
                error = "native deformation selected bone is not in this mesh skin";
                return false;
            }
            const auto index = found->second;
            std::array<float, 4> delta { 0.0f, 0.0f, 0.0f, 1.0f };
            constexpr double halfRadiansPerDegree = 0.0087266462599716478846;
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                translations[index][axis] += static_cast<float>(pose.translation[axis]);
                scales[index][axis] *= static_cast<float>(pose.scale[axis]);
                const auto angle = pose.rotationDegrees[axis] * halfRadiansPerDegree;
                std::array<float, 4> rotation { 0.0f, 0.0f, 0.0f,
                                               static_cast<float>(std::cos(angle)) };
                rotation[axis] = static_cast<float>(std::sin(angle));
                delta = multiplyRotation(rotation, delta);
            }
            rotations[index] = multiplyRotation(rotations[index], delta);
        }
        for (std::size_t index = 0; index < skin->joints().size(); ++index)
        {
            const auto& translation = translations[index];
            const auto& rotation = rotations[index];
            const auto& scale = scales[index];
            const float rotationNorm = rotation[0] * rotation[0]
                + rotation[1] * rotation[1] + rotation[2] * rotation[2]
                + rotation[3] * rotation[3];
            if (! finiteArray (translation) || ! finiteArray (rotation)
                || ! finiteArray (scale) || scale[0] == 0.0f
                || scale[1] == 0.0f || scale[2] == 0.0f
                || ! finite (rotationNorm)
                || std::abs (rotationNorm - 1.0f) > 0.001f)
            {
                error = "native deformation evaluated joint transform is invalid";
                return false;
            }
        }

        std::vector<Matrix> world (skin->joints().size());
        std::vector<std::uint8_t> state (skin->joints().size(), 0);
        std::function<bool(std::size_t)> resolve = [&] (std::size_t index)
        {
            if (state[index] == 2) return true;
            if (state[index] == 1) return false;
            state[index] = 1;
            auto local = compose (translations[index], rotations[index], scales[index]);
            if (source.jointBaseTransforms[index].matrix)
                local = multiply(*source.jointBaseTransforms[index].matrix, local);
            const auto parent = skin->joints()[index].parent();
            if (parent.isValid())
            {
                const auto found = indices.find (parent.value);
                if (found == indices.end() || ! resolve (found->second)) return false;
                world[index] = multiply (world[found->second], local);
            }
            else world[index] = local;
            state[index] = 2;
            return true;
        };
        const auto inverseObject = originalInverse ? *originalInverse : inverseTrs (object->transform);
        const auto& inverseBind = skin->inverseBindMatrixValues();
        if (sampledWorlds) candidateWorlds.reserve(skin->joints().size());
        for (std::size_t index = 0; index < skin->joints().size(); ++index)
        {
            if (! resolve (index))
            {
                error = "native deformation skin hierarchy is not resolvable";
                return false;
            }
            Matrix bind {};
            if (sampledWorlds)
            {
                if (!finiteArray(world[index]))
                { error = "native deformation sampled node world is invalid"; return false; }
                candidateWorlds.push_back({skin->joints()[index].id().value, world[index]});
            }
            std::copy_n (inverseBind.data() + index * 16, 16, bind.data());
            const auto palette = multiply (inverseObject, multiply (world[index], bind));
            std::copy (palette.begin(), palette.end(),
                       candidate.jointPalette.begin() + index * 16);
        }
    }

    bool sampledMorph = false;
    for (const auto& evaluation : snapshot.morphWeights())
    {
        if (source.animationNodeStableId != 0
            && evaluation.animationTarget().value != source.animationNodeStableId) continue;
        if (source.batchMember && evaluation.mesh() != mesh->id()) continue;
        if (evaluation.mesh() != mesh->id() || sampledMorph
            || evaluation.targets().size() != mesh->morphTargets().size()
            || evaluation.values().size() != mesh->morphTargets().size())
        {
            error = "native deformation snapshot morph mesh identity is incompatible";
            return false;
        }
        sampledMorph = true;
        for (std::size_t index = 0; index < mesh->morphTargets().size(); ++index)
        {
            if (evaluation.targets()[index] != mesh->morphTargets()[index].id()
                || ! finite (evaluation.values()[index])
                || std::abs (evaluation.values()[index]) > 16.0f)
            {
                error = "native deformation snapshot morph target identity is incompatible";
                return false;
            }
            if (combinationMode == visualanimation::CombinationMode::WeightedBlend)
                candidate.morphWeights[index] +=
                    (evaluation.values()[index] - candidate.morphWeights[index])
                    * combinationWeight;
            else if (combinationMode == visualanimation::CombinationMode::Multiply)
                candidate.morphWeights[index] *=
                    1.0f + (evaluation.values()[index] - 1.0f) * combinationWeight;
            else if (combinationMode == visualanimation::CombinationMode::Add
                     || combinationMode
                         == visualanimation::CombinationMode::AddAfterImportedAnimation)
                candidate.morphWeights[index] +=
                    evaluation.values()[index] * combinationWeight;
            else candidate.morphWeights[index] = evaluation.values()[index];
        }
    }

    if (pose.morphEnabled)
    {
        // Imported playback's legacy global morph gain still applies to the
        // other targets. The selected target owns its independent final weight.
        if (combinationMode != visualanimation::CombinationMode::WeightedBlend)
            for (std::size_t index = 0; index < mesh->morphTargets().size(); ++index)
                candidate.morphWeights[index] *= combinationWeight;
        candidate.morphWeights[pose.morphTargetIndex] = static_cast<float>(pose.morphWeight);
        // Per-key deltas add to the evaluated weights of their keys, and the sum is clamped
        // to the morph range. A key the mesh lacks is refused, never skipped.
        for (const auto& [keyIndex, delta] : pose.morphWeightDeltas)
        {
            if (keyIndex >= mesh->morphTargets().size() || !std::isfinite(delta)
                || std::abs(delta) > 16.0)
            {
                error = "native deformation morph weight delta is out of range for the mesh";
                return false;
            }
            if (delta == 0.0) continue;
            candidate.morphWeights[keyIndex] = std::clamp(
                candidate.morphWeights[keyIndex] + static_cast<float>(delta), -16.0f, 16.0f);
        }
    }
    output = candidate;
    if (sampledWorlds) *sampledWorlds = std::move(candidateWorlds);
    return true;
}
namespace
{
Matrix sceneLocal(const HarmonicMIDI::grid::SceneTransform3D& transform) noexcept
{
    return compose({transform.translation.x, transform.translation.y, transform.translation.z},
        {transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w},
        {transform.scale.x, transform.scale.y, transform.scale.z});
}

bool inverseAffine(const Matrix& value, Matrix& result) noexcept
{
    if (!finiteArray(value) || value[3] != 0 || value[7] != 0 || value[11] != 0 || value[15] != 1)
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
    return finiteArray(result);
}

bool sameMatrix(const Matrix& a, const Matrix& b) noexcept
{
    for (std::size_t i=0; i<a.size(); ++i)
        if (!finite(a[i]) || !finite(b[i]) || std::abs(a[i]-b[i]) > 1.0e-5f*std::max(1.0f,std::max(std::abs(a[i]),std::abs(b[i]))))
            return false;
    return true;
}

bool decomposeLocal(const Matrix& matrix,std::array<float,3>& translation,
    std::array<float,4>& rotation,std::array<float,3>& scale) noexcept
{
    Matrix inverse;
    if (!inverseAffine(matrix,inverse)) return false;
    for (std::size_t c=0;c<3;++c)
        scale[c]=std::sqrt(matrix[4*c]*matrix[4*c]+matrix[4*c+1]*matrix[4*c+1]+matrix[4*c+2]*matrix[4*c+2]);
    const double determinant=double(matrix[0])*(double(matrix[5])*matrix[10]-double(matrix[9])*matrix[6])
        -double(matrix[4])*(double(matrix[1])*matrix[10]-double(matrix[9])*matrix[2])
        +double(matrix[8])*(double(matrix[1])*matrix[6]-double(matrix[5])*matrix[2]);
    if (determinant<0) scale[0]=-scale[0];
    if (!finiteArray(scale) || scale[0]==0 || scale[1]==0 || scale[2]==0) return false;
    const float a=matrix[0]/scale[0],b=matrix[4]/scale[1],c=matrix[8]/scale[2];
    const float d=matrix[1]/scale[0],e=matrix[5]/scale[1],f=matrix[9]/scale[2];
    const float g=matrix[2]/scale[0],h=matrix[6]/scale[1],i=matrix[10]/scale[2];
    const float trace=a+e+i;
    if (trace>0) {const float q=2*std::sqrt(trace+1);rotation={(h-f)/q,(c-g)/q,(d-b)/q,q/4};}
    else if (a>e && a>i) {const float q=2*std::sqrt(1+a-e-i);rotation={q/4,(b+d)/q,(c+g)/q,(h-f)/q};}
    else if (e>i) {const float q=2*std::sqrt(1+e-a-i);rotation={(b+d)/q,q/4,(f+h)/q,(c-g)/q};}
    else {const float q=2*std::sqrt(1+i-a-e);rotation={(c+g)/q,(f+h)/q,q/4,(d-b)/q};}
    const auto norm=std::sqrt(rotation[0]*rotation[0]+rotation[1]*rotation[1]+rotation[2]*rotation[2]+rotation[3]*rotation[3]);
    if (!finite(norm) || norm==0) return false;
    for (auto& value:rotation) value/=norm;
    translation={matrix[12],matrix[13],matrix[14]};
    return sameMatrix(matrix,compose(translation,rotation,scale));
}

bool sameTexture(const HarmonicMIDI::grid::Visual3DScene& a,
                 HarmonicMIDI::grid::SceneTextureId aid,
                 const HarmonicMIDI::grid::Visual3DScene& b,
                 HarmonicMIDI::grid::SceneTextureId bid) noexcept
{
    if (!aid.isValid() || !bid.isValid()) return aid == bid;
    const auto find = [](const auto& scene, auto id) -> const HarmonicMIDI::grid::SceneTextureRecord* {
        for (std::size_t i=0; i<scene.textureCount; ++i) if (scene.textures[i].id == id) return &scene.textures[i];
        return nullptr;
    };
    const auto* at=find(a,aid); const auto* bt=find(b,bid);
    if (!at || !bt) return false;
    if (at->width!=bt->width || at->height!=bt->height || at->magFilter!=bt->magFilter
        || at->minFilter!=bt->minFilter || at->wrapS!=bt->wrapS || at->wrapT!=bt->wrapT) return false;
    const auto count=std::size_t(at->width)*at->height;
    return std::memcmp(a.textureTexels.data()+at->firstTexel,b.textureTexels.data()+bt->firstTexel,
                       count*sizeof(HarmonicMIDI::grid::SceneTexelRgba8)) == 0;
}

bool sameMaterial(const HarmonicMIDI::grid::Visual3DScene& a,
                  HarmonicMIDI::grid::SceneMaterialId aid,
                  const HarmonicMIDI::grid::Visual3DScene& b,
                  HarmonicMIDI::grid::SceneMaterialId bid) noexcept
{
    if (!aid.isValid() || !bid.isValid()) return aid == bid;
    const auto find = [](const auto& scene, auto id) -> const HarmonicMIDI::grid::SceneMaterialRecord* {
        for (std::size_t i=0; i<scene.materialCount; ++i) if (scene.materials[i].id == id) return &scene.materials[i];
        return nullptr;
    };
    const auto* am=find(a,aid); const auto* bm=find(b,bid);
    if (!am || !bm) return false;
    const std::array<HarmonicMIDI::grid::SceneTextureId,5> ats {am->baseColorTexture,am->metallicRoughnessTexture,
        am->normalTexture,am->occlusionTexture,am->emissiveTexture};
    const std::array<HarmonicMIDI::grid::SceneTextureId,5> bts {bm->baseColorTexture,bm->metallicRoughnessTexture,
        bm->normalTexture,bm->occlusionTexture,bm->emissiveTexture};
    for (std::size_t i=0; i<ats.size(); ++i) if (!sameTexture(a,ats[i],b,bts[i])) return false;
    const auto sameVec=[](const auto& a,const auto& b) {return a.x==b.x && a.y==b.y && a.z==b.z;};
    return sameVec(am->baseColor,bm->baseColor) && sameVec(am->emissive,bm->emissive)
        && am->opacity==bm->opacity && am->metallic==bm->metallic && am->roughness==bm->roughness
        && am->normalScale==bm->normalScale && am->occlusionStrength==bm->occlusionStrength
        && am->alphaMode==bm->alphaMode && am->alphaCutoff==bm->alphaCutoff
        && am->doubleSided==bm->doubleSided;
}

bool sameBase(const NativeDeformationJointBaseTransform& a, const NativeDeformationJointBaseTransform& b) noexcept
{
    return a.skin==b.skin && a.joint==b.joint && a.translation==b.translation && a.rotation==b.rotation
        && a.scale==b.scale && a.matrix==b.matrix;
}

bool sameMemberData(const NativeDeformationScene& source, const NativeDeformationScene& original) noexcept
{
    return source.sourceStableId==original.sourceStableId && source.deformationStableId==original.deformationStableId
        && source.structuralRevision==original.structuralRevision
        && source.evaluationRevision==original.evaluationRevision && source.clip==original.clip
        && source.mesh==original.mesh && source.skin==original.skin && source.deformation==original.deformation
        && source.animationNodeStableId==original.animationNodeStableId && source.morphBaseWeights==original.morphBaseWeights
        && source.retainDeformedGeometry==original.retainDeformedGeometry
        && source.jointBaseTransforms.size()==original.jointBaseTransforms.size()
        && std::equal(source.jointBaseTransforms.begin(),source.jointBaseTransforms.end(),original.jointBaseTransforms.begin(),sameBase);
}
} // namespace

std::shared_ptr<const NativeDeformationCompositionOwner> NativeDeformationCompositionOwner::admit(
    visualmodelassetpayload::PayloadPtr payload,
    std::shared_ptr<const videohelper::gltf::GlbStaticMeshDocument> document,
    std::shared_ptr<const videohelper::gltf::GlbAnimationDocument> animation,
    std::optional<std::size_t> selectedScene, std::uint64_t selectedMesh,
    std::shared_ptr<const NativeDeformationScene> imported,
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural,
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed,
    const std::vector<NativeDeformationCompositionMapping>& mappings,
    std::uint64_t projectGeneration, std::uint64_t helperGeneration, std::string& error,
    std::uint64_t evaluationRevision)
{
    // The single leaf lives on this stack frame; the owner keeps no leaf list.
    Leaf leaf;
    leaf.input.payload=std::move(payload); leaf.input.document=std::move(document);
    leaf.input.animation=std::move(animation); leaf.input.selectedScene=selectedScene;
    leaf.input.selectedMesh=selectedMesh;
    if (imported) leaf.input.scene=imported->scene;
    leaf.input.imported=std::move(imported);
    return admitLeaves(&leaf,1,std::move(procedural),std::move(composed),mappings,
        projectGeneration,helperGeneration,error,evaluationRevision,nullptr);
}

std::shared_ptr<const NativeDeformationCompositionOwner> NativeDeformationCompositionOwner::admit(
    std::vector<NativeDeformationCompositionLeaf> inputs,
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural,
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed,
    const std::vector<NativeDeformationCompositionMapping>& mappings,
    std::uint64_t projectGeneration, std::uint64_t helperGeneration, std::string& error,
    std::uint64_t evaluationRevision)
{
    error.clear();
    if (inputs.size()<2 || inputs.size()>HarmonicMIDI::grid::scenecomposition::kMaximumImportedLeaves)
    { error="native composition multi-leaf owner requires two to eight leaves"; return {}; }
    std::vector<Leaf> leaves(inputs.size());
    for (std::size_t index=0; index<inputs.size(); ++index) leaves[index].input=std::move(inputs[index]);
    auto* first=leaves.data();
    return admitLeaves(first,leaves.size(),std::move(procedural),std::move(composed),mappings,
        projectGeneration,helperGeneration,error,evaluationRevision,&leaves);
}

std::shared_ptr<const NativeDeformationCompositionOwner> NativeDeformationCompositionOwner::admitLeaves(
    Leaf* leaves, std::size_t leafCount,
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural,
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed,
    const std::vector<NativeDeformationCompositionMapping>& mappings,
    std::uint64_t projectGeneration, std::uint64_t helperGeneration, std::string& error,
    std::uint64_t evaluationRevision, std::vector<Leaf>* retained)
{
    using namespace HarmonicMIDI::grid;
    using Kind=NativeDeformationCompositionMapping::Kind;
    error.clear();
    const auto refuse=[&](const char* message) -> std::shared_ptr<const NativeDeformationCompositionOwner> {
        error=message; return {};
    };
    const bool single=leafCount==1;
    // Each leaf keeps its own node bound. Several leaves address leaf k's
    // node r as k * kImportedLeafIdStride + r, so each also stays below it.
    const std::size_t leafNodeCapacity=single?std::size_t(4096):std::size_t(scenecomposition::kImportedLeafIdStride);
    std::size_t identity=leafCount;
    for (std::size_t k=0; k<leafCount && identity==leafCount; ++k) if (leaves[k].input.animation) identity=k;
    const auto validLeaf=[&](std::size_t k) {
        const auto& leaf=leaves[k].input;
        const auto& payload=leaf.payload; const auto& document=leaf.document;
        const auto& imported=leaf.imported;
        if (!payload || payload->bytes().empty() || payload->bytes().size()>visualmodelassetpayload::kMaxPayloadBytes
            || payload->key().sourceByteSize!=payload->bytes().size() || !document
            || (leaf.selectedScene && *leaf.selectedScene!=document->selectedScene)
            || document->selectedScene>=document->scenes.size()
            || document->nodes.size()>leafNodeCapacity
            || !scenecomposition::importedLeafCountFits(leafCount,document->nodes.size()))
            return false;
        if (!leaf.animation)
            return !imported && leaf.scene && validateVisual3DScene(*leaf.scene).valid();
        const auto& animation=leaf.animation;
        const auto& head=leaves[identity].input.imported;
        return imported && imported->scene && imported->deformation && !imported->compositionOwner
            && imported->sourceStableId!=0 && imported->deformationStableId!=0 && imported->structuralRevision!=0
            && imported->evaluationRevision==evaluationRevision && evaluationRevision>=imported->structuralRevision
            && imported->clip.isValid() && animation->deformation==imported->deformation
            && animation->renderBindings.size()<=64 && animation->clips.size()<=256
            && leaf.scene==imported->scene && head && imported->structuralRevision==head->structuralRevision
            && validateVisual3DScene(*imported->scene).valid();
    };
    bool admissible=identity<leafCount && leafCount<=scenecomposition::kMaximumImportedLeaves
        && procedural && composed && projectGeneration!=0 && helperGeneration!=0 && evaluationRevision!=0;
    std::size_t importedObjects=0;
    for (std::size_t k=0; admissible && k<leafCount; ++k)
    {
        admissible=validLeaf(k);
        if (admissible) importedObjects+=leaves[k].input.scene->objectCount;
    }
    if (!admissible || mappings.size()!=composed->objectCount || mappings.empty() || mappings.size()>64
        || importedObjects+procedural->objectCount!=composed->objectCount
        || !validateVisual3DScene(*procedural).valid() || !validateVisual3DScene(*composed).valid())
        return refuse("native composition owner inputs or aggregate bounds are incompatible");
    // Source-loop envelope: every object scan, hierarchy walk, vertex/index
    // comparison, five texture scans/comparisons, and per-member legacy pose
    // validation. Existing immutable payload/doc/scene allocations stay with
    // their released owners; this cap covers added IR, rigid assets and scratch.
    // Several leaves add their retained leaf records and one hierarchy walk
    // per leaf at that leaf's node bound.
    const auto n=mappings.size();
    std::size_t bytes=2*singleLeafBytes()+4096*sizeof(bool)+12288*sizeof(bool)+n*(sizeof(NativeDeformationScene)+sizeof(visualdeformation::DeformationAsset)
            +sizeof(visualdeformation::Skin)+sizeof(visualdeformation::Joint)+sizeof(visualdeformation::Mesh)
            +sizeof(visualdeformation::JointWeightSet)+sizeof(NativeDeformationJointBaseTransform)
            +sizeof(std::vector<std::uint32_t>)+sizeof(std::vector<float>))
        +2*4096*4*(sizeof(std::uint32_t)+sizeof(float))+2*n*sizeof(NativeDeformationFrameData)
        +leafListBytes(leafCount);
    std::size_t work=leafCount*(leafNodeCapacity*leafNodeCapacity*64+leafNodeCapacity*512)+n*n*256
        +composed->vertexCount*256+composed->indexCount*32
        +n*5*Visual3DScene::kMaxTextureTexels*sizeof(SceneTexelRgba8);
    if (bytes>kMaximumLogicalBytes || work>kMaximumLogicalWork)
        return refuse("native composition incremental owner storage or work exceeds capacity");
    NativeDeformationCompositionOwner candidate;
    const auto& head=leaves[identity].input;
    candidate.payload_=head.payload; candidate.document_=head.document; candidate.animation_=head.animation;
    candidate.selectedScene_=head.selectedScene; candidate.selectedMesh_=head.selectedMesh; candidate.imported_=head.imported;
    candidate.procedural_=std::move(procedural); candidate.composed_=std::move(composed);
    candidate.revision_=candidate.imported_->structuralRevision;
    candidate.evaluationRevision_=evaluationRevision;
    candidate.projectGeneration_=projectGeneration; candidate.helperGeneration_=helperGeneration;
    for (std::size_t k=0; k<leafCount; ++k)
    {
        leaves[k].nodeCount=leaves[k].input.document->nodes.size();
        leaves[k].nodeOffset=candidate.importedNodeCount_;
        candidate.importedNodeCount_+=leaves[k].nodeCount;
    }
    candidate.recordCount_=n;
    candidate.logicalBytes_=bytes; candidate.logicalWork_=work;
    // Imported mappings name their leaf through the stride identity. Procedural
    // mappings belong to no leaf. One leaf keeps raw node identities.
    std::array<std::size_t,64> leafOf {};
    std::array<std::uint64_t,64> rawNode {};
    for (std::size_t i=0; i<n; ++i)
    {
        const auto& mapping=mappings[i];
        leafOf[i]=leafCount;
        if (mapping.kind!=Kind::Imported) continue;
        if (single) {leafOf[i]=0; rawNode[i]=mapping.importedNodeStableId; continue;}
        const auto decoded=mapping.importedNodeStableId<0x80000000u
            ? scenecomposition::decodeImportedLeafId(static_cast<std::uint32_t>(mapping.importedNodeStableId),leafCount)
            : std::nullopt;
        if (!decoded) return refuse("native composition imported node mapping is incomplete");
        leafOf[i]=decoded->leaf; rawNode[i]=decoded->raw;
    }
    const auto recordFor=[&](Kind kind,std::size_t leaf,SceneObjectId id) {
        for (std::size_t i=0; i<n; ++i)
            if (mappings[i].kind==kind && leafOf[i]==leaf && mappings[i].sourceObject==id) return i;
        return n;
    };
    const auto composedRecord=[&](SceneObjectId id) {
        for (std::size_t i=0; i<n; ++i) if (mappings[i].composedObject==id) return i;
        return n;
    };
    const auto sourceScene=[&](std::size_t i) -> const Visual3DScene& {
        return mappings[i].kind==Kind::Imported ? *leaves[leafOf[i]].input.scene : *candidate.procedural_;
    };
    std::array<bool,Visual3DScene::kMaxVertices> vertices {};
    std::array<bool,Visual3DScene::kMaxIndices> indices {};
    for (std::size_t i=0; i<n; ++i)
    {
        const auto& mapping=mappings[i];
        const auto& original=sourceScene(i);
        const auto* a=findObject(original,mapping.sourceObject);
        const auto* b=findObject(*candidate.composed_,mapping.composedObject);
        if ((mapping.kind!=Kind::Imported && mapping.kind!=Kind::Procedural) || !a || !b
            || recordFor(mapping.kind,leafOf[i],mapping.sourceObject)!=i || composedRecord(mapping.composedObject)!=i
            || a->vertexCount!=b->vertexCount || a->indexCount!=b->indexCount
            || std::memcmp(&a->transform,&b->transform,sizeof(a->transform))!=0
            || !sameMaterial(original,a->material,*candidate.composed_,b->material)
            || (mapping.kind==Kind::Procedural && mapping.importedNodeStableId!=0))
            return refuse("native composition mapping changes source topology, material or local transform");
        if (mapping.kind==Kind::Imported)
        {
            if (rawNode[i]==0 || rawNode[i]>leaves[leafOf[i]].nodeCount)
                return refuse("native composition imported node mapping is incomplete");
            const auto node=std::size_t(rawNode[i]-1);
            const auto first=visualimportedsceneidentity::objectId(node,(a->id.value-1)%visualimportedsceneidentity::kObjectPrimitiveStride);
            const auto transform=visualimportedsceneidentity::nodeTransformObjectId(node);
            if ((!first || a->id!=*first) && (!transform || a->id!=*transform))
                return refuse("native composition imported object does not match its original node identity");
        }
        for (std::size_t v=0; v<a->vertexCount; ++v)
        {
            const auto target=b->firstVertex+v;
            if (vertices[target] || std::memcmp(&original.vertices[a->firstVertex+v],
                    &candidate.composed_->vertices[target],sizeof(SceneVertex))!=0)
                return refuse("native composition vertex ranges overlap or change rest geometry");
            vertices[target]=true;
        }
        for (std::size_t j=0; j<a->indexCount; ++j)
        {
            const auto target=b->firstIndex+j;
            if (indices[target] || original.indices[a->firstIndex+j]!=candidate.composed_->indices[target])
                return refuse("native composition index ranges overlap or change rest geometry");
            indices[target]=true;
        }
        auto& record=candidate.records_[i]; record.mapping=mapping; record.local=sceneLocal(a->transform);
        record.parent=b->parent.isValid()?composedRecord(b->parent):64;
        record.originalParent=a->parent.isValid()?recordFor(mapping.kind,leafOf[i],a->parent):64;
        // A source root may gain a parent only from another source: procedural
        // versus imported, or another imported leaf.
        if ((b->parent.isValid() && record.parent==n) || (a->parent.isValid() && record.originalParent==n)
            || (a->parent.isValid() && record.parent!=record.originalParent)
            || (!a->parent.isValid() && b->parent.isValid() && mappings[record.parent].kind==mapping.kind
                && leafOf[record.parent]==leafOf[i]))
            return refuse("native composition permits only cross-source root parent additions");
    }
    if (std::size_t(std::count(vertices.begin(),vertices.end(),true))!=candidate.composed_->vertexCount
        || std::size_t(std::count(indices.begin(),indices.end(),true))!=candidate.composed_->indexCount)
        return refuse("native composition mapping omits scene topology");
    // Iterative fixed scratch makes both cycle detection and the work bound
    // independent of recursion and of input object order.
    for (bool original : {true,false})
    {
        std::array<bool,64> resolved {};
        for (std::size_t pass=0; pass<n; ++pass)
            for (std::size_t i=0; i<n; ++i)
            {
                auto& r=candidate.records_[i]; const auto parent=original?r.originalParent:r.parent;
                if (resolved[i] || (parent!=64 && !resolved[parent])) continue;
                auto& world=original?r.originalWorld:r.staticWorld;
                world=parent==64?r.local:multiply(original?candidate.records_[parent].originalWorld:
                    candidate.records_[parent].staticWorld,r.local);
                Matrix inverse;
                if (!inverseAffine(world,inverse)) return refuse("native composition hierarchy has a singular or nonfinite world");
                if (!original) r.inverseStaticWorld=inverse;
                resolved[i]=true;
            }
        for (std::size_t i=0; i<n; ++i) if (!resolved[i]) return refuse("native composition hierarchy has a cycle");
    }
    for (std::size_t i=0; i<n; ++i)
        if (mappings[i].kind==Kind::Imported
            && !sameMatrix(candidate.records_[i].originalWorld,
                leaves[leafOf[i]].input.document->nodes[rawNode[i]-1].worldTransform))
            return refuse("native composition imported rest basis disagrees with admitted node hierarchy");
    for (std::size_t k=0; k<leafCount; ++k)
    {
        const auto& nodes=leaves[k].input.document->nodes;
        for (std::size_t i=0; i<nodes.size(); ++i)
        {
            const auto& node=nodes[i];
            Matrix inverse;
            if ((node.parent && *node.parent>=nodes.size())
                || !inverseAffine(node.localTransform,inverse) || !inverseAffine(node.worldTransform,inverse)
                || !sameMatrix(node.worldTransform,node.parent
                    ? multiply(nodes[*node.parent].worldTransform,node.localTransform):node.localTransform))
                return refuse("native composition imported local hierarchy is inconsistent");
            auto ancestor=node.parent;
            for (std::size_t step=0; ancestor; ++step)
            {
                if (step>=nodes.size() || *ancestor>=nodes.size())
                    return refuse("native composition imported local hierarchy is cyclic");
                ancestor=nodes[*ancestor].parent;
            }
        }
    }
    for (std::size_t k=0; k<leafCount; ++k)
        if (leaves[k].input.imported && leaves[k].input.imported->draws.size()>64)
            return refuse("native composition imported draw count exceeds capacity");
    for (std::size_t i=0; i<n; ++i)
    {
        const auto& r=candidate.records_[i];
        // Static leaves have no native draw; they become rigid members below.
        if (r.mapping.kind==Kind::Imported && !leaves[leafOf[i]].input.animation) continue;
        const auto& original=sourceScene(i);
        const auto* object=findObject(original,r.mapping.sourceObject);
        if (object->vertexCount==0) continue;
        if (r.mapping.kind==Kind::Imported)
        {
            const auto& leaf=leaves[leafOf[i]].input;
            const auto& source=leaf.imported;
            const auto& draws=source->draws;
            std::shared_ptr<const NativeDeformationScene> draw;
            if (draws.empty() && source->object==object->id) draw=source;
            for (const auto& value:draws) if (value && value->object==object->id)
            { if (draw) return refuse("native composition imported member is duplicated"); draw=value; }
            const auto* mesh=draw && draw->deformation?draw->deformation->findMesh(draw->mesh):nullptr;
            if (!draw || !draw->draws.empty() || draw->compositionOwner || draw->scene!=source->scene
                || draw->sourceStableId!=source->sourceStableId
                || draw->deformationStableId!=source->deformationStableId
                || draw->structuralRevision!=candidate.revision_ || draw->evaluationRevision!=evaluationRevision
                || draw->clip!=source->clip
                || draw->retainDeformedGeometry!=source->retainDeformedGeometry
                || (leaf.selectedMesh!=0 && draw->mesh.value!=leaf.selectedMesh)
                || draw->deformation!=source->deformation || draw->animationNodeStableId!=rawNode[i]
                || !mesh || mesh->vertexCount()!=object->vertexCount || mesh->jointWeightSets().size()>1
                || mesh->morphTargets().size()>kNativeDeformationMaxMorphTargets || draw->jointBaseTransforms.size()>64)
                return refuse("native composition imported member loses deformation identity");
            const auto skinId=draw->skin.value_or(mesh->skin());
            const auto* skin=skinId.isValid()?draw->deformation->findSkin(skinId):nullptr;
            if ((skinId.isValid() && (!skin || skin->joints().empty() || skin->joints().size()>64
                    || skin->joints().size()!=draw->jointBaseTransforms.size() || mesh->jointWeightSets().size()!=1))
                || (!skinId.isValid() && (!draw->jointBaseTransforms.empty() || !mesh->jointWeightSets().empty()))
                || (!draw->morphBaseWeights.empty() && draw->morphBaseWeights.size()!=mesh->morphTargets().size()))
                return refuse("native composition original skin or morph bounds are incompatible");
            for (const auto& target:mesh->morphTargets())
                if (target.hasTangentDeltas()) return refuse("native composition tangent morph deltas exceed the released subset");
            const auto binding=std::find_if(leaf.animation->renderBindings.begin(),leaf.animation->renderBindings.end(),
                [&](const auto& b){return b.nodeIndex+1==rawNode[i] && b.mesh==draw->mesh;});
            if (binding==leaf.animation->renderBindings.end() || binding->skin!=draw->skin.value_or(mesh->skin())
                || binding->morphBaseWeights!=draw->morphBaseWeights || binding->jointBaseTransforms.size()!=draw->jointBaseTransforms.size())
                return refuse("native composition imported binding is foreign");
            for (std::size_t j=0; j<binding->jointBaseTransforms.size(); ++j)
            {
                const auto& a=binding->jointBaseTransforms[j]; const auto& b=draw->jointBaseTransforms[j];
                if (a.skin!=b.skin || a.joint!=b.joint || a.translation!=b.translation || a.rotation!=b.rotation
                    || a.scale!=b.scale || a.matrix!=b.matrix)
                    return refuse("native composition imported base transform is foreign");
                const auto norm=b.rotation[0]*b.rotation[0]+b.rotation[1]*b.rotation[1]
                    +b.rotation[2]*b.rotation[2]+b.rotation[3]*b.rotation[3];
                Matrix inverse;
                const auto local=b.matrix?multiply(*b.matrix,compose(b.translation,b.rotation,b.scale)):
                    compose(b.translation,b.rotation,b.scale);
                if (!skin || b.skin!=skin->id() || b.joint!=skin->joints()[j].id()
                    || !finiteArray(b.translation) || !finiteArray(b.rotation) || !finiteArray(b.scale)
                    || !finite(norm) || std::abs(norm-1)>0.001f || !inverseAffine(local,inverse))
                    return refuse("native composition imported base joint is singular, nonfinite or foreign");
            }
            for (auto weight:draw->morphBaseWeights)
                if (!finite(weight) || std::abs(weight)>16)
                    return refuse("native composition original morph base weight exceeds its finite bounds");
            candidate.members_[candidate.memberCount_++]={i,std::move(draw)};
        }
    }
    // All structural and aggregate admission precedes rigid-asset allocations.
    // Procedural objects and static-leaf objects share the rigid identity skin.
    // Its 64-bit ID keeps the two kinds apart: a procedural ID is
    // 0x8000000000000000 + object; a static leaf ID is 0xC000000000000000 +
    // (leaf << 32) + object. The object ID is 32 bits wide and the leaf is
    // below 8, so the fields never overlap or carry into the tag bits.
    static_assert(sizeof(SceneObjectId::value)==sizeof(std::uint32_t),
        "rigid identity packs a 32-bit object ID below the leaf field");
    static_assert(scenecomposition::kMaximumImportedLeaves<=(std::uint64_t(1)<<29),
        "rigid identity leaf field stays below the static tag");
    for (std::size_t i=0; i<n; ++i)
    {
        const auto& r=candidate.records_[i];
        const bool staticImported=r.mapping.kind==Kind::Imported && !leaves[leafOf[i]].input.animation;
        if (r.mapping.kind!=Kind::Procedural && !staticImported) continue;
        const auto& original=sourceScene(i);
        const auto* object=findObject(original,r.mapping.sourceObject);
        if (object->vertexCount==0) continue;
        std::vector<std::uint32_t> joints(object->vertexCount*4,0);
        std::vector<float> weights(object->vertexCount*4,0);
        for (std::size_t v=0; v<object->vertexCount; ++v) weights[v*4]=1;
        const std::uint64_t identity=staticImported
            ? 0xC000000000000000ull+(std::uint64_t(leafOf[i])<<32)+r.mapping.sourceObject.value
            : 0x8000000000000000ull+r.mapping.sourceObject.value;
        const visualdeformation::JointView joint {{identity},{}};
        const Matrix identityMatrix {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        const visualdeformation::SkinView skin {{identity},&joint,1,identityMatrix.data(),identityMatrix.size()};
        const visualdeformation::JointWeightSetView influence {joints.data(),joints.size(),weights.data(),weights.size()};
        const visualdeformation::MeshView mesh {{identity},object->vertexCount,skin.id,&influence,1,nullptr,0};
        visualdeformation::Limits limits; limits.maxSkins=1; limits.maxMeshes=1; limits.maxJointsPerSkin=1;
        limits.maxTotalJoints=1; limits.maxVerticesPerMesh=4096; limits.maxTotalVertices=4096;
        limits.maxInfluenceSetsPerMesh=1; limits.maxMorphTargetsPerMesh=0; limits.maxTotalMorphTargets=0;
        limits.maxTotalAccessorValues=4096*8+16;
        auto asset=visualdeformation::DeformationAsset::create({&skin,1,&mesh,1},limits,error);
        if (!asset) return {};
        auto draw=std::make_shared<NativeDeformationScene>();
        draw->sourceStableId=candidate.imported_->sourceStableId; draw->deformationStableId=candidate.imported_->deformationStableId;
        draw->structuralRevision=candidate.revision_; draw->evaluationRevision=evaluationRevision;
        draw->clip=candidate.imported_->clip;
        draw->mesh=mesh.id; draw->skin=skin.id; draw->object=object->id;
        draw->scene=staticImported?leaves[leafOf[i]].input.scene:candidate.procedural_;
        if (candidate.imported_->logicalCompositionRetention)
        {
            struct RetainedAsset final
            {
                std::shared_ptr<const void> reservation;
                std::shared_ptr<const visualdeformation::DeformationAsset> asset;
            };
            auto retainedAsset=std::make_shared<RetainedAsset>(RetainedAsset {candidate.imported_->logicalCompositionRetention,asset});
            asset={std::move(retainedAsset),asset.get()};
        }
        draw->deformation=std::move(asset); draw->batchMember=true;
        draw->logicalCompositionRetention=candidate.imported_->logicalCompositionRetention;
        draw->retainDeformedGeometry=candidate.imported_->retainDeformedGeometry;
        draw->jointBaseTransforms.push_back({skin.id,joint.id,{0,0,0},{0,0,0,1},{1,1,1},std::nullopt});
        candidate.members_[candidate.memberCount_++]={i,std::move(draw)};
    }
    if (retained) candidate.leaves_=std::move(*retained);
    return std::shared_ptr<const NativeDeformationCompositionOwner>(new NativeDeformationCompositionOwner(std::move(candidate)));
}

bool NativeDeformationCompositionOwner::matches(const NativeDeformationScene& source) const noexcept
{
    if (source.compositionOwner.get()!=this || source.scene!=composed_ || source.structuralRevision!=revision_
        || source.evaluationRevision!=evaluationRevision_) return false;
    // The batch and every single-leaf member carry the identity leaf's source,
    // deformation and clip. Members of other leaves carry their own leaf's,
    // which sameMemberData authenticates against the admitted member.
    const bool identity=source.sourceStableId==imported_->sourceStableId
        && source.deformationStableId==imported_->deformationStableId && source.clip==imported_->clip;
    if (!source.draws.empty())
    {
        if (!identity || source.batchMember || source.draws.size()!=memberCount_ || source.deformation!=imported_->deformation) return false;
        std::array<bool,64> found {};
        for (const auto& draw:source.draws)
        {
            if (!draw || !draw->batchMember || !draw->draws.empty() || !matches(*draw)) return false;
            for (std::size_t i=0; i<memberCount_; ++i) if (draw->object==memberObject(i))
            { if (found[i]) return false; found[i]=true; }
        }
        return true;
    }
    if (!source.batchMember || (leaves_.empty() && !identity)) return false;
    for (std::size_t i=0; i<memberCount_; ++i)
        if (source.object==memberObject(i)) return sameMemberData(source,*members_[i].source);
    return false;
}

std::shared_ptr<const NativeDeformationCompositionFrame> NativeDeformationCompositionFrame::admit(
    std::shared_ptr<const NativeDeformationCompositionOwner> owner,
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation,
    std::shared_ptr<const visualanimation::Sample> sceneAnimation,
    visualdeformation::RationalFrameTime time, std::uint64_t projectGeneration,
    std::uint64_t helperGeneration, std::string& error)
{
    // The single sample lives on this stack frame; the frame keeps no leaf list.
    NativeDeformationCompositionLeafSample sample;
    sample.deformation=std::move(deformation); sample.sceneAnimation=std::move(sceneAnimation);
    return admitLeaves(std::move(owner),&sample,1,nullptr,time,projectGeneration,helperGeneration,error);
}

std::shared_ptr<const NativeDeformationCompositionFrame> NativeDeformationCompositionFrame::admit(
    std::shared_ptr<const NativeDeformationCompositionOwner> owner,
    std::vector<NativeDeformationCompositionLeafSample> samples,
    visualdeformation::RationalFrameTime time, std::uint64_t projectGeneration,
    std::uint64_t helperGeneration, std::string& error)
{
    error.clear();
    if (!owner || owner->leafCount()<2 || samples.size()!=owner->leafCount())
    { error="native composition multi-leaf frame requires one sample per owner leaf"; return {}; }
    const auto* first=samples.data();
    const auto count=samples.size();
    return admitLeaves(std::move(owner),first,count,&samples,time,projectGeneration,helperGeneration,error);
}

std::shared_ptr<const NativeDeformationCompositionFrame> NativeDeformationCompositionFrame::admitLeaves(
    std::shared_ptr<const NativeDeformationCompositionOwner> owner,
    const NativeDeformationCompositionLeafSample* samples, std::size_t sampleCount,
    std::vector<NativeDeformationCompositionLeafSample>* retained,
    visualdeformation::RationalFrameTime time, std::uint64_t projectGeneration,
    std::uint64_t helperGeneration, std::string& error)
{
    using Kind=NativeDeformationCompositionMapping::Kind;
    error.clear();
    const auto refuse=[&](const char* message) -> std::shared_ptr<const NativeDeformationCompositionFrame> {
        error=message; return {};
    };
    if (!owner || sampleCount!=owner->leafCount() || time.rateNumerator==0 || time.rateDenominator==0
        || projectGeneration!=owner->projectGeneration_ || helperGeneration!=owner->helperGeneration_)
        return refuse("native composition frame owner, time or generation is incompatible");
    const auto& authority=*owner;
    const auto leafCount=sampleCount;
    const auto leafDocument=[&](std::size_t leaf) -> const videohelper::gltf::GlbStaticMeshDocument& {
        return authority.leaves_.empty() ? *authority.document_ : *authority.leaves_[leaf].input.document;
    };
    const auto leafAnimation=[&](std::size_t leaf) -> const videohelper::gltf::GlbAnimationDocument& {
        return authority.leaves_.empty() ? *authority.animation_ : *authority.leaves_[leaf].input.animation;
    };
    const auto leafClip=[&](std::size_t leaf) {
        return authority.leaves_.empty() ? authority.imported_->clip : authority.leaves_[leaf].input.imported->clip;
    };
    const auto leafOffset=[&](std::size_t leaf) -> std::size_t {
        return authority.leaves_.empty() ? std::size_t(0) : authority.leaves_[leaf].nodeOffset;
    };
    // Every animated leaf samples its own clip at its own clip time, all for
    // the same presentation frame. Static leaves carry no sample.
    for (std::size_t leaf=0; leaf<leafCount; ++leaf)
    {
        const auto& deformation=samples[leaf].deformation;
        if (authority.staticLeaf(leaf))
        {
            if (deformation || samples[leaf].sceneAnimation)
                return refuse("native composition frame owner, time or generation is incompatible");
            continue;
        }
        if (!deformation || !(deformation->time()==time) || deformation->revision()!=authority.revision_
            || deformation->clip()!=leafClip(leaf) || !sceneMotionSnapshotFits(deformation))
            return refuse("native composition frame owner, time or generation is incompatible");
        // Clip, skin and joint IDs repeat across GLBs, so each leaf sample must
        // also name its leaf's exact payload content.
        std::array<std::uint8_t,32> digest {};
        if (!authority.leaves_.empty()
            && (!visualdeformation::sha256Digest(authority.leaves_[leaf].input.payload->key().contentSha256,digest)
                || digest!=deformation->sourceContentSha256()))
            return refuse("native composition leaf snapshot is not from its leaf's exact payload");
    }
    std::array<const visualanimation::Clip*,HarmonicMIDI::grid::scenecomposition::kMaximumImportedLeaves> clips {};
    for (std::size_t leaf=0; leaf<leafCount; ++leaf)
    {
        if (authority.staticLeaf(leaf)) continue;
        const auto& animation=leafAnimation(leaf);
        const auto& deformation=*samples[leaf].deformation;
        const auto clip=std::find_if(animation.clips.begin(),animation.clips.end(),
            [&](const auto& named){return named.clip && named.clip->id()==deformation.clip();});
        if (clip==animation.clips.end() || clip->clip->tracks().size()>4096)
            return refuse("native composition frame has no exact admitted clip");
        clips[leaf]=clip->clip.get();
    }
    const auto nodes=authority.importedNodeCount_;
    // Three node tables, two object tables, two sets of post uploads including
    // Metal's 256-byte alignment, both retained samples and legacy preparation
    // scratch. Source work counts bounded loop bodies and scalar operations;
    // allocator, library and compiler/driver-private overhead is unknown.
    // Admission precedes all new vectors and frame construction. Several
    // leaves add their retained sample list and one snapshot each.
    std::size_t bytes=2*singleLeafBytes()+nodes*(3*sizeof(Matrix)+4*sizeof(std::uint8_t))
        +2*64*sizeof(Matrix)+2*64*256+2*64*sizeof(NativeDeformationFrameData)
        +2*nodes*sizeof(NativeSceneMotionSample::ImportedNodeWorldMatrix)
        +64*(64*sizeof(NativeDeformationJointBaseTransform)+4*sizeof(Matrix)+sizeof(std::uint8_t))
        +18*sizeof(float)+leafListBytes(retained?leafCount:1);
    const auto addBytes=[&](std::size_t count,std::size_t stride) {
        if (count>(NativeDeformationCompositionOwner::kMaximumLogicalBytes-bytes)/stride) return false;
        bytes+=count*stride; return true;
    };
    if (bytes>NativeDeformationCompositionOwner::kMaximumLogicalBytes)
        return refuse("native composition frame storage exceeds capacity");
    for (std::size_t leaf=0; leaf<leafCount; ++leaf)
    {
        if (authority.staticLeaf(leaf)) continue;
        const auto& deformation=samples[leaf].deformation;
        const auto& sceneAnimation=samples[leaf].sceneAnimation;
        if (!addBytes(1,sizeof(*deformation))
            || !addBytes(deformation->jointTransforms().capacity(),sizeof(visualdeformation::JointTransformEvaluation))
            || !addBytes(deformation->morphWeights().capacity(),sizeof(visualdeformation::MorphWeightEvaluation)))
            return refuse("native composition frame storage exceeds capacity");
        for (const auto& morph:deformation->morphWeights())
            if (!addBytes(1,sizeof(visualdeformation::MorphWeightSample))
                || !addBytes(morph.values().capacity(),sizeof(float))
                || !addBytes(morph.targets().capacity(),sizeof(visualdeformation::MorphTargetId)))
                return refuse("native composition retained morph storage exceeds capacity");
        if (sceneAnimation)
        {
            const auto& tracks=clips[leaf]->tracks();
            if (sceneAnimation->clip()!=deformation->clip() || sceneAnimation->requestedTimeSeconds()!=deformation->requestedTimeSeconds()
                || sceneAnimation->sampleTimeSeconds()!=deformation->sampleTimeSeconds() || sceneAnimation->playback()!=deformation->playback()
                || sceneAnimation->tracks().size()!=tracks.size()
                || !addBytes(1,sizeof(*sceneAnimation)) || !addBytes(sceneAnimation->tracks().capacity(),sizeof(visualanimation::TrackSample)))
                return refuse("native composition scene animation is foreign or exceeds capacity");
            for (std::size_t i=0; i<sceneAnimation->tracks().size(); ++i)
            {
                const auto& a=sceneAnimation->tracks()[i]; const auto& b=tracks[i];
                if (a.track()!=b.id() || a.target()!=b.target() || a.channel()!=b.channel() || a.values().size()!=b.valueWidth()
                    || !std::all_of(a.values().begin(),a.values().end(),finite) || !addBytes(a.values().capacity(),sizeof(float)))
                    return refuse("native composition scene animation changes admitted channel identity");
            }
        }
    }
    if (bytes>NativeDeformationCompositionOwner::kMaximumLogicalBytes/2
        || authority.logicalBytes_>NativeDeformationCompositionOwner::kMaximumLogicalBytes-2*bytes)
        return refuse("native composition retained current and preceding frames exceed capacity");
    // A joint scan can visit every retained node and each legacy pose can scan
    // every sampled joint. Matrix products/inverses have fixed scalar loops.
    // Reserve the complete enabled source invocation in either backend, with
    // eight morph targets and four nonzero influences. One unit is a scalar
    // arithmetic/comparison/Boolean/branch step or an explicit scalar
    // read/write/initialization/constructor/projection/copy lane visit.
    // Arithmetic operand access belongs to that operation, not another visit.
    // Count every written constructor lane and product component, including
    // the discarded fourth component of a float4 product. No compiler elision
    // is assumed. Matrix-vector products cost 16 multiplies + 12 adds for 4x4,
    // or 9 multiplies + 6 adds for 3x3; cross/dot/length cost 9/5/6 operations.
    // A bounded loop has one counter initialization, n+1 compare/branch pairs
    // and n increment/write pairs. Conditional vectors reserve the larger arm.
    constexpr std::size_t loop4=1+2*5+2*4, loop8=1+3*9+2*8, loop16=1+2*17+2*16;
    // Morph's loop also reads its uniform bound at each comparison.
    // Each morph visits five delta count-read/address/init steps, two weight
    // read/init steps and two vectors: two address adds, three loads,
    // constructor lanes, multiplies, adds and accumulator writes.
    constexpr std::size_t morphShaderWork=loop8+8*(5+2+2*(2+3+3+3+3+3));
    // Length and its initialization, comparison/branch, then either three
    // divides or three fallback constructor lanes, and three result writes.
    constexpr std::size_t normalizeShaderWork=6+1+1+1+3+3;
    // Three normal component reads, nine cofactor multiplies, six vector adds,
    // three divides and three normal writes. Reflection is compare/branch,
    // selected sign, multiply and handedness write.
    constexpr std::size_t inverseNormalShaderWork=3+9+6+3+3, reflectionShaderWork=5;
    // GL prefix: invocation read/init, range read/compare/branch, base/output
    // address/init, 16 copy iterations with two addresses/load/store, then
    // position and normal address/load/constructor/init. The selector is
    // read/compare/branch. Tangent/handedness loads occur only in its enabled arm.
    constexpr std::size_t glPrefixWork=2+3+2+3+loop16+16*4+(2+3+3+3)+(3+3+3+3);
    constexpr std::size_t glTangentLoadWork=3*(1+1)+3+3+1+1+1;
    // GL skin: joint-count selection, 16 zero constructor/init lanes, influence
    // multiply/init, four joint/weight address/read/init visits and joint-count
    // uniform read, two comparisons, Boolean and two short-circuit branches.
    // jointMatrix has first multiply/init, 16 address/read/constructor lanes
    // and call/return. Its weighted 16-lane sum has multiply/add/write per lane.
    constexpr std::size_t glJointMatrixWork=2+16*3+2;
    constexpr std::size_t glSkinWork=3+16*2+2+loop4+4*(3+3+6+glJointMatrixWork+16*3);
    // Position float4 constructor/product/projection/write; three crosses
    // each visit six indexed lanes and six projection lanes plus cross/init;
    // determinant has three indexed/projection lanes plus dot/init. Tangent
    // has nine mat3 constructor lanes, its product and three writes.
    constexpr std::size_t glSkinBasisWork=(4+28+3+3)+3*(6+6+9+3)+(3+3+5+1)
        +inverseNormalShaderWork+(9+15+3)+reflectionShaderWork;
    // GL post: two full matrix read/constructor/init copies, point product,
    // normal and tangent mat3 products, tangent normalization and four output
    // addresses/component reads/writes including determinant read/multiply.
    // Common final normal normalization and six stores remain included.
    constexpr std::size_t glPostWork=2*(16+16+16)+(4+28+3+3)+2*(9+15+3)
        +normalizeShaderWork+3*3+(1+1+1+1);
    constexpr std::size_t glOutputWork=normalizeShaderWork+6+1+5*2;
    constexpr std::size_t glShaderWork=glPrefixWork+morphShaderWork+3+glTangentLoadWork
        +glSkinWork+glSkinBasisWork+glPostWork+glOutputWork;
    // Metal prefix: range read/compare/branch, full input read/init, two packed
    // vector read/constructor/init conversions and full output copy. Tangent
    // reads/converts four lanes, projects/initializes three; handedness read/init.
    constexpr std::size_t metalPrefixWork=3+16*2+2*(3+3+3)+16;
    constexpr std::size_t metalTangentLoadWork=4+4+3+3+2;
    // Metal skin: joint-count selection, nine matrix/three position zero
    // constructor/init lanes, four influence address/init, weight/joint
    // read/init and zero-weight compare/branch visits. Each palette
    // reads/initializes all 16 lanes;
    // position uses float4 constructor/product/projection and weight/add/write;
    // linear sum reads/projects/constructs/multiplies/adds/writes all nine lanes.
    constexpr std::size_t metalSkinWork=3+9*2+3*2+loop4
        +4*((3+2+2+2)+16*2+(4+28+3+3+3+3)+9*6)+3;
    // Metal cofactor/determinant have indexed reads without GL's xyz projection.
    // Tangent multiplies its existing mat3 directly. Post uses three full
    // float4 matrix products, including normal/tangent fourth components.
    constexpr std::size_t metalSkinBasisWork=3*(6+9+3)+(3+5+1)
        +inverseNormalShaderWork+(15+3)+reflectionShaderWork;
    constexpr std::size_t metalPostWork=3*(16+4+28+3+3)+2*normalizeShaderWork
        +(1+1+4+4+4);
    // Final packed position, inherited normal selector with enabled lane copy
    // and packed conversion/store, UV read/write and output-offset read/add
    // plus full destination copy/store. Disabled normalization is not executed
    // in this schedule; its source expression remains unchanged in Metal.
    constexpr std::size_t metalOutputWork=(3+3)+(1+1+1+3+3+3)+2*2+(1+1+16+16);
    constexpr std::size_t metalShaderWork=metalPrefixWork+morphShaderWork+3+metalTangentLoadWork
        +metalSkinWork+metalSkinBasisWork+metalPostWork+metalOutputWork;
    constexpr std::size_t enabledShaderWork=glShaderWork>metalShaderWork?glShaderWork:metalShaderWork;
    // Both dispatch 64 lanes per group. At most 63 padding lanes per member
    // execute only the range guard and return. GL has invocation read/init,
    // count read, comparison, branch and return; Metal has only the last four.
    constexpr std::size_t paddedShaderWork=63*6;
    // This is a private enabled reservation. The inherited selector still runs
    // on disabled submissions; no complete legacy scalar-work ledger is claimed.
    // Typed host scan: nine clears; four influence visits with 16 fixed steps
    // plus nine cells bounded by 16 address/read/update/loop steps; 27 cofactor
    // ops, five determinant ops, nine divide/finite pairs, nine finite visits
    // and 16 per-vertex control steps. Batch preflight and member dispatch each
    // perform this scan. The factor two covers current and preceding dispatch.
    constexpr std::size_t skinHostWork=9+4*(16+9*16)+27+5+9*2+9*2+16;
    std::size_t work=authority.memberCount_*4096*64*128+64*64*256
        +authority.composed_->vertexCount*2*(enabledShaderWork+2*skinHostWork)
        +authority.memberCount_*2*paddedShaderWork;
    for (std::size_t leaf=0; leaf<leafCount; ++leaf)
    {
        const auto count=leafDocument(leaf).nodes.size();
        work+=count*count*64+(clips[leaf]?clips[leaf]->tracks().size():std::size_t(0))*count*4;
    }
    if (work>NativeDeformationCompositionOwner::kMaximumLogicalWork)
        return refuse("native composition frame reconstruction or dispatch work exceeds capacity");
    std::vector<Matrix> raw(nodes);
    for (std::size_t leaf=0; leaf<leafCount; ++leaf)
    {
        const auto& document=leafDocument(leaf);
        const auto count=document.nodes.size(), offset=leafOffset(leaf);
        // A static leaf holds its admitted rest hierarchy.
        if (authority.staticLeaf(leaf))
        {
            for (std::size_t i=0;i<count;++i) raw[offset+i]=document.nodes[i].worldTransform;
            continue;
        }
        const auto& deformation=*samples[leaf].deformation;
        const auto& sceneAnimation=samples[leaf].sceneAnimation;
        std::vector<Matrix> locals(count);
        std::vector<std::uint8_t> covered(count,0), animated(count,0),sampledChannels(count,0),resolvedNodes(count,0);
        for (std::size_t i=0;i<count;++i) locals[i]=document.nodes[i].localTransform;
        for (const auto& track:clips[leaf]->tracks())
            if (track.channel()!=visualanimation::Channel::MorphWeights)
            {
                if (track.target().value==0 || track.target().value>count)
                    return refuse("native composition channel references a missing imported node");
                animated[track.target().value-1]=1;
            }
        for (std::size_t m=0; m<authority.memberCount_; ++m)
        {
            const auto& member=authority.members_[m]; const auto& r=authority.records_[member.record];
            if (r.mapping.kind!=Kind::Imported || authority.memberRigid(m) || authority.memberLeaf(m)!=leaf) continue;
            Matrix inverseOriginal;
            if (!inverseAffine(r.originalWorld,inverseOriginal)) return refuse("native composition original world is not invertible");
            NativeDeformationFrameData ignored;
            NativeSceneMotionSample::ImportedNodeWorldMatrices sampled;
            if (!prepareOriginalNativeDeformationFrame(*member.source,deformation,ignored,error,&sampled,&inverseOriginal)) return {};
            for (const auto& value:sampled)
            {
                if (value.nodeStableId==0 || value.nodeStableId>count)
                    return refuse("native composition sampled ancestor is not an imported node");
                const auto node=std::size_t(value.nodeStableId-1);
                if (covered[node] && !sameMatrix(raw[offset+node],value.world))
                    return refuse("native composition sampled ancestors disagree across members");
                raw[offset+node]=value.world; covered[node]=1;
            }
        }
        // Canonical joint worlds include combination and selected bone controls.
        // Fill uncovered scene channels from the already admitted sample, then
        // reconstruct static descendants through those evaluated parent worlds.
        // Never resample a clip or overwrite a joint-covered node with raw channels.
        if (sceneAnimation)
            for (std::size_t node=0;node<count;++node)
            {
                if (covered[node] || !animated[node]) continue;
                std::array<float,3> t,scale;
                std::array<float,4> rotation;
                if (!decomposeLocal(locals[node],t,rotation,scale))
                    return refuse("native composition scene channel base is not a finite invertible TRS");
                for (const auto& track:sceneAnimation->tracks())
                {
                    if (track.target().value!=node+1 || track.channel()==visualanimation::Channel::MorphWeights) continue;
                    const auto weight=static_cast<float>(deformation.combinationWeight());
                    const auto mode=deformation.combinationMode();
                    if (track.channel()==visualanimation::Channel::Rotation)
                    {
                        const std::array<float,4> value {track.values()[0],track.values()[1],track.values()[2],track.values()[3]};
                        const std::array<float,4> identity {0,0,0,1};
                        if (mode==visualanimation::CombinationMode::WeightedBlend) rotation=weightedRotation(rotation,value,weight);
                        else if (mode==visualanimation::CombinationMode::Replace) rotation=value;
                        else {
                            const auto delta=weightedRotation(identity,value,weight);
                            rotation=mode==visualanimation::CombinationMode::Add?multiplyRotation(delta,rotation):multiplyRotation(rotation,delta);
                        }
                    }
                    else
                    {
                        const std::array<float,3> value {track.values()[0],track.values()[1],track.values()[2]};
                        auto& base=track.channel()==visualanimation::Channel::Translation?t:scale;
                        const std::array<float,3> identity=track.channel()==visualanimation::Channel::Translation
                            ?std::array<float,3>{0,0,0}:std::array<float,3>{1,1,1};
                        if (mode==visualanimation::CombinationMode::WeightedBlend) base=weightedBlend(base,value,weight);
                        else if (mode==visualanimation::CombinationMode::Multiply) base=multiplyRelative(base,value,identity,weight);
                        else if (mode==visualanimation::CombinationMode::Replace) base=value;
                        else base=addRelative(base,value,identity,weight);
                    }
                    sampledChannels[node]=1;
                }
                const auto norm=rotation[0]*rotation[0]+rotation[1]*rotation[1]+rotation[2]*rotation[2]+rotation[3]*rotation[3];
                if (!finiteArray(t) || !finiteArray(rotation) || !finiteArray(scale) || !finite(norm) || std::abs(norm-1)>0.001f)
                    return refuse("native composition sampled scene TRS is nonfinite or has an unnormalized rotation");
                locals[node]=compose(t,rotation,scale);
            }
        for (std::size_t i=0;i<count;++i)
        {
            if (animated[i] && !covered[i] && !sampledChannels[i])
                return refuse("native composition lacks authenticated animated ancestor channels");
            resolvedNodes[i]=covered[i];
            Matrix inverse;
            if (covered[i] && !inverseAffine(raw[offset+i],inverse))
                return refuse("native composition sampled joint world is singular or nonfinite");
        }
        for (std::size_t pass=0;pass<count;++pass)
            for (std::size_t i=0;i<count;++i)
            {
                const auto parent=document.nodes[i].parent;
                if (resolvedNodes[i] || (parent && !resolvedNodes[*parent])) continue;
                raw[offset+i]=parent?multiply(raw[offset+*parent],locals[i]):locals[i];
                Matrix inverse;
                if (!inverseAffine(raw[offset+i],inverse)) return refuse("native composition sampled descendant is singular or nonfinite");
                resolvedNodes[i]=1;
            }
        for (auto resolved:resolvedNodes)
            if (!resolved) return refuse("native composition animated ancestor coverage is incomplete");
    }
    const auto identityLeaf=authority.identityLeaf();
    NativeDeformationCompositionFrame candidate;
    candidate.deformation_=samples[identityLeaf].deformation;
    candidate.sceneAnimation_=samples[identityLeaf].sceneAnimation; candidate.time_=time; candidate.logicalBytes_=bytes;
    candidate.logicalWork_=work;
    // One leaf keeps raw one-based node identities; several leaves publish
    // leaf k's node r as k * kImportedLeafIdStride + r.
    candidate.rawWorlds_.reserve(nodes);
    for (std::size_t leaf=0; leaf<leafCount; ++leaf)
    {
        const auto count=leafDocument(leaf).nodes.size(), offset=leafOffset(leaf);
        const std::uint64_t base=authority.leaves_.empty()?std::uint64_t(0)
            :std::uint64_t(leaf)*HarmonicMIDI::grid::scenecomposition::kImportedLeafIdStride;
        for (std::size_t node=0;node<count;++node) candidate.rawWorlds_.push_back({base+node+1,raw[offset+node]});
    }
    std::array<Matrix,64> evaluated {}, external {};
    std::array<bool,64> resolved {};
    const Matrix identity {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    for (std::size_t pass=0; pass<authority.recordCount_; ++pass)
        for (std::size_t i=0; i<authority.recordCount_; ++i)
        {
            const auto& r=authority.records_[i];
            if (resolved[i] || (r.parent!=64 && !resolved[r.parent])) continue;
            if (r.mapping.kind==Kind::Imported)
            {
                // E(t) is the external composed root placement. Imported W(t)
                // is kept separate because legacy skin already consumes it.
                external[i]=r.originalParent!=64 ? external[r.originalParent]
                    : r.parent!=64 ? evaluated[r.parent] : identity;
                const auto leaf=authority.recordLeaf(i);
                if (authority.staticLeaf(leaf)) evaluated[i]=multiply(external[i],r.originalWorld);
                else
                {
                    const auto node=std::size_t((authority.leaves_.empty() ? r.mapping.importedNodeStableId
                        : r.mapping.importedNodeStableId%HarmonicMIDI::grid::scenecomposition::kImportedLeafIdStride)-1);
                    Matrix inverseRestNode;
                    if (!inverseAffine(leafDocument(leaf).nodes[node].worldTransform,inverseRestNode))
                        return refuse("native composition imported rest node is singular");
                    evaluated[i]=multiply(external[i],multiply(raw[leafOffset(leaf)+node],multiply(inverseRestNode,r.originalWorld)));
                }
            }
            else
                evaluated[i]=r.parent==64?r.local:multiply(evaluated[r.parent],r.local);
            Matrix inverse;
            if (!inverseAffine(evaluated[i],inverse)) return refuse("native composition evaluated descendant is singular or nonfinite");
            resolved[i]=true;
        }
    for (std::size_t i=0; i<authority.recordCount_; ++i)
        if (!resolved[i]) return refuse("native composition evaluated hierarchy is incomplete");
    for (std::size_t m=0; m<authority.memberCount_; ++m)
    {
        const auto& member=authority.members_[m]; const auto& r=authority.records_[member.record];
        // Rigid members (procedural and static-leaf objects) carry an identity
        // palette, so their post places the whole evaluated object.
        const bool rigid=authority.memberRigid(m);
        auto point=multiply(r.inverseStaticWorld,evaluated[member.record]);
        const auto* mesh=member.source->deformation->findMesh(member.source->mesh);
        if (!rigid && r.mapping.kind==Kind::Imported && member.source->skin.value_or(mesh->skin()).isValid())
            point=multiply(r.inverseStaticWorld,multiply(external[member.record],r.originalWorld));
        Matrix inverse;
        if (!inverseAffine(point,inverse)) return refuse("native composition post transform is singular or nonfinite");
        auto& post=candidate.posts_[m]; post.control_={1,1,rigid?2.0f:1.0f,0};
        post.point_=point; post.normal_=identity;
        for (std::size_t column=0; column<3; ++column)
            for (std::size_t row=0; row<3; ++row) post.normal_[column*4+row]=inverse[row*4+column];
        const double determinant=double(point[0])*(double(point[5])*point[10]-double(point[9])*point[6])
            -double(point[4])*(double(point[1])*point[10]-double(point[9])*point[2])
            +double(point[8])*(double(point[1])*point[6]-double(point[5])*point[2]);
        post.normal_[15]=determinant<0?-1.0f:1.0f;
    }
    if (retained) candidate.leaves_=std::move(*retained);
    candidate.owner_=std::move(owner);
    return std::shared_ptr<const NativeDeformationCompositionFrame>(new NativeDeformationCompositionFrame(std::move(candidate)));
}

bool NativeDeformationCompositionFrame::matches(const NativeDeformationScene& source,
    const visualdeformation::AnimationDeformationSnapshot& snapshot) const noexcept
{
    return owner_ && deformation_.get()==&snapshot && owner_->matches(source)
        && snapshot.time()==time_ && snapshot.revision()==owner_->revision_;
}

bool prepareNativeDeformationFrame(const NativeDeformationScene& source,
    const visualdeformation::AnimationDeformationSnapshot& snapshot,
    const std::shared_ptr<const NativeDeformationCompositionFrame>& composition,
    NativeDeformationFrameData& output, std::string& error,
    NativeSceneMotionSample::ImportedNodeWorldMatrices* sampledWorlds)
{
    if (!composition)
        return prepareNativeDeformationFrame(source,snapshot,output,error,sampledWorlds);
    error.clear();
    if (!composition->matches(source,snapshot))
    { error="native composition frame has a foreign or stale exact owner"; return false; }
    if (sampledWorlds) sampledWorlds->clear();
    NativeDeformationFrameData candidate;
    NativeSceneMotionSample::ImportedNodeWorldMatrices worlds;
    const auto& owner=*composition->owner_;
    if (!source.draws.empty())
    {
        for (const auto& draw:source.draws)
        {
            NativeDeformationFrameData value;
            if (!prepareNativeDeformationFrame(*draw,snapshot,composition,value,error,nullptr)) return false;
        }
    }
    else
    {
        std::size_t index=0;
        while (index<owner.memberCount_ && owner.memberObject(index)!=source.object) ++index;
        if (index==owner.memberCount_) {error="native composition member is missing";return false;}
        const auto& member=owner.members_[index]; const auto& r=owner.records_[member.record];
        // Each animated leaf member takes its own leaf's snapshot, so weight and
        // pose apply per member. Rigid members keep an identity palette.
        const bool imported=!owner.memberRigid(index);
        Matrix inverseOriginal;
        if (!inverseAffine(imported?r.originalWorld:Matrix {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1},inverseOriginal)
            || !prepareOriginalNativeDeformationFrame(*member.source,composition->memberSnapshot(index),candidate,error,
                nullptr,&inverseOriginal)) return false;
        // Reflection uses the actual weighted skin determinant in both native
        // shaders. Reject an undefined handedness before uploading or dispatching;
        // individual invertible joints can still blend to a singular linear map.
        if (candidate.jointCount)
        {
            const auto* mesh=member.source->deformation->findMesh(member.source->mesh);
            if (!mesh || mesh->jointWeightSets().size()!=1)
            { error="native composition skin influence topology changed"; return false; }
            const auto& influences=mesh->jointWeightSets()[0];
            const auto& indices=influences.jointIndices(); const auto& weights=influences.weights();
            if (indices.size()!=mesh->vertexCount()*4 || weights.size()!=indices.size())
            { error="native composition skin influence ranges changed"; return false; }
            for (std::size_t vertex=0;vertex<mesh->vertexCount();++vertex)
            {
                std::array<float,9> linear {};
                for (std::size_t slot=0;slot<4;++slot)
                {
                    const auto influence=vertex*4+slot, joint=std::size_t(indices[influence]);
                    const auto weight=weights[influence];
                    if (weight==0) continue;
                    if (joint>=candidate.jointCount)
                    { error="native composition skin joint is outside its palette"; return false; }
                    for (std::size_t column=0;column<3;++column)
                        for (std::size_t row=0;row<3;++row)
                            linear[column*3+row]+=weight*candidate.jointPalette[joint*16+column*4+row];
                }
                // These float cofactors match the uploaded weighted matrix and
                // the explicit cross-product normal transform in both shaders.
                std::array<float,9> cofactors {
                    linear[4]*linear[8]-linear[5]*linear[7],
                    linear[5]*linear[6]-linear[3]*linear[8],
                    linear[3]*linear[7]-linear[4]*linear[6],
                    linear[7]*linear[2]-linear[8]*linear[1],
                    linear[8]*linear[0]-linear[6]*linear[2],
                    linear[6]*linear[1]-linear[7]*linear[0],
                    linear[1]*linear[5]-linear[2]*linear[4],
                    linear[2]*linear[3]-linear[0]*linear[5],
                    linear[0]*linear[4]-linear[1]*linear[3]};
                const float determinant=linear[0]*cofactors[0]+linear[1]*cofactors[1]+linear[2]*cofactors[2];
                if (!finiteArray(linear) || !finite(determinant) || determinant==0)
                { error="native composition weighted skin is singular or nonfinite"; return false; }
                for (auto& coefficient:cofactors) coefficient/=determinant;
                if (!finiteArray(cofactors))
                { error="native composition weighted skin normal inverse is nonfinite"; return false; }
            }
        }
        candidate.compositionPost=composition->posts_[index];
    }
    if (sampledWorlds) worlds=composition->rawWorlds_;
    output=std::move(candidate);
    if (sampledWorlds) *sampledWorlds=std::move(worlds);
    return true;
}

bool prepareNativeDeformationFrame (
    const NativeDeformationScene& source,
    const visualdeformation::AnimationDeformationSnapshot& snapshot,
    NativeDeformationFrameData& output, std::string& error,
    NativeSceneMotionSample::ImportedNodeWorldMatrices* sampledWorlds)
{
    if (source.compositionOwner)
    { error = "native composition requires its exact typed frame"; return false; }
    return prepareOriginalNativeDeformationFrame(source, snapshot, output, error, sampledWorlds);
}
} // namespace arbitgpu
