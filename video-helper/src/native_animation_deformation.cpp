#include "gpu_backend/backend.h"

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

bool prepareNativeDeformationFrame (
    const NativeDeformationScene& source,
    const visualdeformation::AnimationDeformationSnapshot& snapshot,
    NativeDeformationFrameData& output,
    std::string& error)
{
    using namespace visualdeformation;
    error.clear();
    NativeDeformationFrameData candidate;
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
    if (source.scene->objectCount != 1 || source.scene->materialCount != 1
        || source.scene->lightCount != 1 || source.scene->cameraCount != 1)
    {
        error = "native deformation admits one object, material, light, and camera";
        return false;
    }
    const auto* object = findObject (*source.scene, source.object);
    const auto* mesh = source.deformation->findMesh (source.mesh);
    if (object == nullptr || mesh == nullptr || object != &source.scene->objects[0]
        || object->parent.isValid() || object->firstVertex != 0
        || object->vertexCount != source.scene->vertexCount
        || mesh->vertexCount() != object->vertexCount
        || mesh->vertexCount() == 0 || mesh->vertexCount() > kNativeDeformationMaxVertices)
    {
        error = "native deformation object and mesh ranges are incompatible";
        return false;
    }
    if (source.scene->materials[0].opacity != 1.0f
        || source.scene->lights[0].kind
            != HarmonicMIDI::grid::SceneLightKind::Directional)
    {
        error = "native deformation scene is outside the opaque fixture subset";
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
    if (mesh->skin().isValid())
    {
        skin = source.deformation->findSkin (mesh->skin());
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
            const auto local = compose (translations[index], rotations[index], scales[index]);
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
        const auto inverseObject = inverseTrs (object->transform);
        const auto& inverseBind = skin->inverseBindMatrixValues();
        for (std::size_t index = 0; index < skin->joints().size(); ++index)
        {
            if (! resolve (index))
            {
                error = "native deformation skin hierarchy is not resolvable";
                return false;
            }
            Matrix bind {};
            std::copy_n (inverseBind.data() + index * 16, 16, bind.data());
            const auto palette = multiply (inverseObject, multiply (world[index], bind));
            std::copy (palette.begin(), palette.end(),
                       candidate.jointPalette.begin() + index * 16);
        }
    }

    bool sampledMorph = false;
    for (const auto& evaluation : snapshot.morphWeights())
    {
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

    output = candidate;
    return true;
}
} // namespace arbitgpu
