#include "../../shared/VisualDeformationContract.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>

namespace visualdeformation
{
namespace
{
Limits boundedLimits (const Limits& requested) noexcept
{
    const Limits hard;
    Limits result;
    result.maxSkins = std::min(requested.maxSkins, hard.maxSkins);
    result.maxMeshes = std::min(requested.maxMeshes, hard.maxMeshes);
    result.maxJointsPerSkin = std::min(requested.maxJointsPerSkin, hard.maxJointsPerSkin);
    result.maxTotalJoints = std::min(requested.maxTotalJoints, hard.maxTotalJoints);
    result.maxVerticesPerMesh = std::min(requested.maxVerticesPerMesh, hard.maxVerticesPerMesh);
    result.maxTotalVertices = std::min(requested.maxTotalVertices, hard.maxTotalVertices);
    result.maxInfluenceSetsPerMesh = std::min(requested.maxInfluenceSetsPerMesh,
                                               hard.maxInfluenceSetsPerMesh);
    result.maxMorphTargetsPerMesh = std::min(requested.maxMorphTargetsPerMesh,
                                              hard.maxMorphTargetsPerMesh);
    result.maxTotalMorphTargets = std::min(requested.maxTotalMorphTargets,
                                            hard.maxTotalMorphTargets);
    result.maxTotalAccessorValues = std::min(requested.maxTotalAccessorValues,
                                              hard.maxTotalAccessorValues);
    result.weightSumTolerance = std::isfinite(requested.weightSumTolerance)
        && requested.weightSumTolerance >= 0.0f
        ? std::min(requested.weightSumTolerance, hard.weightSumTolerance) : 0.0f;
    result.maxInverseBindElementMagnitude = std::isfinite(requested.maxInverseBindElementMagnitude)
        && requested.maxInverseBindElementMagnitude >= 0.0f
        ? std::min(requested.maxInverseBindElementMagnitude,
                   hard.maxInverseBindElementMagnitude) : 0.0f;
    result.maxMorphDeltaMagnitude = std::isfinite(requested.maxMorphDeltaMagnitude)
        && requested.maxMorphDeltaMagnitude >= 0.0f
        ? std::min(requested.maxMorphDeltaMagnitude, hard.maxMorphDeltaMagnitude) : 0.0f;
    result.maxMorphWeightMagnitude = std::isfinite(requested.maxMorphWeightMagnitude)
        && requested.maxMorphWeightMagnitude >= 0.0f
        ? std::min(requested.maxMorphWeightMagnitude, hard.maxMorphWeightMagnitude) : 0.0f;
    return result;
}

bool multiplyWithinSize (std::size_t left, std::size_t right,
                         std::size_t& result) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool addWithin (std::size_t value, std::size_t& total, std::size_t limit) noexcept
{
    if (value > limit || total > limit - value)
        return false;
    total += value;
    return true;
}

bool valuesAreFiniteAndBounded (const float* values, std::size_t count,
                                float magnitude) noexcept
{
    if (values == nullptr)
        return false;
    for (std::size_t index = 0; index < count; ++index)
        if (! std::isfinite(values[index]) || std::abs(values[index]) > magnitude)
            return false;
    return true;
}

bool validateSkin (const SkinView& source, const Limits& limits,
                   std::size_t& totalJoints, std::size_t& totalValues,
                   std::string& error)
{
    if (! source.id.isValid())
    {
        error = "skin has an invalid stable identity";
        return false;
    }
    if (source.jointCount == 0 || source.jointCount > limits.maxJointsPerSkin
        || ! addWithin(source.jointCount, totalJoints, limits.maxTotalJoints))
    {
        error = "skin joint capacity exceeded";
        return false;
    }
    if (source.joints == nullptr)
    {
        error = "skin has missing joint storage";
        return false;
    }

    std::size_t expectedMatrixValues = 0;
    if (! multiplyWithinSize(source.jointCount, 16, expectedMatrixValues)
        || source.inverseBindValueCount != expectedMatrixValues)
    {
        error = "inverse bind matrix accessor cardinality does not match the joint count";
        return false;
    }
    if (! addWithin(expectedMatrixValues, totalValues, limits.maxTotalAccessorValues))
    {
        error = "deformation accessor value capacity exceeded";
        return false;
    }
    if (! valuesAreFiniteAndBounded(source.inverseBindMatrixValues,
                                    expectedMatrixValues,
                                    limits.maxInverseBindElementMagnitude))
    {
        error = "inverse bind matrix contains a nonfinite or out-of-range value";
        return false;
    }

    std::unordered_map<std::uint64_t, std::size_t> jointIndices;
    jointIndices.reserve(source.jointCount);
    for (std::size_t index = 0; index < source.jointCount; ++index)
    {
        const auto id = source.joints[index].id;
        if (! id.isValid())
        {
            error = "skin joint has an invalid stable identity";
            return false;
        }
        if (! jointIndices.emplace(id.value, index).second)
        {
            error = "skin has a duplicate joint identity";
            return false;
        }
    }

    constexpr auto noParent = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> parentIndices(source.jointCount, noParent);
    for (std::size_t index = 0; index < source.jointCount; ++index)
    {
        const auto parent = source.joints[index].parent;
        if (! parent.isValid())
            continue;
        const auto found = jointIndices.find(parent.value);
        if (found == jointIndices.end())
        {
            error = "skin joint parent is outside the skin";
            return false;
        }
        if (found->second == index)
        {
            error = "skin joint hierarchy contains a cycle";
            return false;
        }
        parentIndices[index] = found->second;
    }

    std::vector<std::uint8_t> state(source.jointCount, 0);
    for (std::size_t start = 0; start < source.jointCount; ++start)
    {
        auto current = start;
        while (current != noParent && state[current] == 0)
        {
            state[current] = 1;
            current = parentIndices[current];
        }
        if (current != noParent && state[current] == 1)
        {
            error = "skin joint hierarchy contains a cycle";
            return false;
        }
        current = start;
        while (current != noParent && state[current] == 1)
        {
            state[current] = 2;
            current = parentIndices[current];
        }
    }
    return true;
}

const SkinView* findSkinView (const AssetView& source, SkinId id) noexcept
{
    for (std::size_t index = 0; index < source.skinCount; ++index)
        if (source.skins[index].id == id)
            return &source.skins[index];
    return nullptr;
}

bool validateJointWeights (const MeshView& source, const SkinView& skin,
                           const Limits& limits, std::size_t& totalValues,
                           std::string& error)
{
    if (source.jointWeightSetCount == 0
        || source.jointWeightSetCount > limits.maxInfluenceSetsPerMesh)
    {
        error = "mesh joint-weight set capacity exceeded";
        return false;
    }
    if (source.jointWeightSets == nullptr)
    {
        error = "skinned mesh has missing joint-weight set storage";
        return false;
    }

    std::size_t expectedValues = 0;
    if (! multiplyWithinSize(source.vertexCount, 4, expectedValues))
    {
        error = "joint-weight accessor cardinality overflow";
        return false;
    }
    for (std::size_t setIndex = 0; setIndex < source.jointWeightSetCount; ++setIndex)
    {
        const auto& set = source.jointWeightSets[setIndex];
        if (set.jointIndexCount != expectedValues || set.weightCount != expectedValues)
        {
            error = "joint and weight accessor cardinalities do not match the vertex count";
            return false;
        }
        if (set.jointIndices == nullptr || set.weights == nullptr)
        {
            error = "skinned mesh has missing joint or weight accessor storage";
            return false;
        }
        if (! addWithin(expectedValues, totalValues, limits.maxTotalAccessorValues)
            || ! addWithin(expectedValues, totalValues, limits.maxTotalAccessorValues))
        {
            error = "deformation accessor value capacity exceeded";
            return false;
        }
        for (std::size_t value = 0; value < expectedValues; ++value)
        {
            if (set.jointIndices[value] >= skin.jointCount)
            {
                error = "joint accessor index is outside the bound skin";
                return false;
            }
            const auto weight = set.weights[value];
            if (! std::isfinite(weight) || weight < 0.0f)
            {
                error = "skin weight is negative or nonfinite";
                return false;
            }
        }
    }

    for (std::size_t vertex = 0; vertex < source.vertexCount; ++vertex)
    {
        double sum = 0.0;
        for (std::size_t setIndex = 0; setIndex < source.jointWeightSetCount; ++setIndex)
            for (std::size_t component = 0; component < 4; ++component)
                sum += source.jointWeightSets[setIndex].weights[vertex * 4 + component];
        if (! std::isfinite(sum) || sum <= 0.0
            || std::abs(sum - 1.0) > static_cast<double>(limits.weightSumTolerance))
        {
            error = "skin weights for a vertex are not normalized";
            return false;
        }
    }
    return true;
}

bool validateMorphTargets (const MeshView& source, const Limits& limits,
                           std::size_t& totalTargets, std::size_t& totalValues,
                           std::string& error)
{
    if (source.morphTargetCount == 0)
        return true;
    if (source.morphTargetCount > limits.maxMorphTargetsPerMesh
        || ! addWithin(source.morphTargetCount, totalTargets,
                       limits.maxTotalMorphTargets))
    {
        error = "mesh morph target capacity exceeded";
        return false;
    }
    if (source.morphTargets == nullptr)
    {
        error = "mesh has missing morph target storage";
        return false;
    }

    std::size_t expectedValues = 0;
    if (! multiplyWithinSize(source.vertexCount, 3, expectedValues))
    {
        error = "morph target accessor cardinality overflow";
        return false;
    }
    std::unordered_set<std::uint64_t> targetIds;
    targetIds.reserve(source.morphTargetCount);
    for (std::size_t targetIndex = 0; targetIndex < source.morphTargetCount; ++targetIndex)
    {
        const auto& target = source.morphTargets[targetIndex];
        if (! target.id.isValid())
        {
            error = "morph target has an invalid stable identity";
            return false;
        }
        if (! targetIds.emplace(target.id.value).second)
        {
            error = "mesh has a duplicate morph target identity";
            return false;
        }
        if (target.positionDeltaCount != expectedValues
            || (target.normalDeltaCount != 0 && target.normalDeltaCount != expectedValues)
            || (target.tangentDeltaCount != 0 && target.tangentDeltaCount != expectedValues))
        {
            error = "morph target accessor cardinality does not match the vertex count";
            return false;
        }
        if (! valuesAreFiniteAndBounded(target.positionDeltas, expectedValues,
                                        limits.maxMorphDeltaMagnitude)
            || (target.normalDeltaCount != 0
                && ! valuesAreFiniteAndBounded(target.normalDeltas, expectedValues,
                                               limits.maxMorphDeltaMagnitude))
            || (target.tangentDeltaCount != 0
                && ! valuesAreFiniteAndBounded(target.tangentDeltas, expectedValues,
                                               limits.maxMorphDeltaMagnitude)))
        {
            error = "morph target contains a missing, nonfinite, or out-of-range delta";
            return false;
        }
        if ((target.normalDeltaCount == 0 && target.normalDeltas != nullptr)
            || (target.tangentDeltaCount == 0 && target.tangentDeltas != nullptr))
        {
            error = "empty morph target accessor has unexpected storage";
            return false;
        }
        if (! addWithin(target.positionDeltaCount, totalValues,
                        limits.maxTotalAccessorValues)
            || ! addWithin(target.normalDeltaCount, totalValues,
                           limits.maxTotalAccessorValues)
            || ! addWithin(target.tangentDeltaCount, totalValues,
                           limits.maxTotalAccessorValues))
        {
            error = "deformation accessor value capacity exceeded";
            return false;
        }
    }
    return true;
}

bool validateMesh (const MeshView& source, const AssetView& asset,
                   const Limits& limits, std::size_t& totalVertices,
                   std::size_t& totalTargets, std::size_t& totalValues,
                   std::string& error)
{
    if (! source.id.isValid())
    {
        error = "deformed mesh has an invalid stable identity";
        return false;
    }
    if (source.vertexCount == 0 || source.vertexCount > limits.maxVerticesPerMesh
        || ! addWithin(source.vertexCount, totalVertices, limits.maxTotalVertices))
    {
        error = "deformed mesh vertex capacity exceeded";
        return false;
    }
    if (! source.skin.isValid())
    {
        if (source.jointWeightSetCount != 0 || source.jointWeightSets != nullptr)
        {
            error = "unskinned mesh declares joint-weight accessors";
            return false;
        }
    }
    else
    {
        const auto* skin = findSkinView(asset, source.skin);
        if (skin == nullptr)
        {
            error = "mesh-to-skin binding references an unknown stable skin identity";
            return false;
        }
        if (! validateJointWeights(source, *skin, limits, totalValues, error))
            return false;
    }
    if (! validateMorphTargets(source, limits, totalTargets, totalValues, error))
        return false;
    if (! source.skin.isValid() && source.morphTargetCount == 0)
    {
        error = "deformed mesh declares neither a skin nor morph targets";
        return false;
    }
    return true;
}
} // namespace

std::shared_ptr<const DeformationAsset> DeformationAsset::create (
    const AssetView& source, const Limits& requestedLimits, std::string& error)
{
    error.clear();
    const auto limits = boundedLimits(requestedLimits);
    if (source.skinCount > limits.maxSkins)
    {
        error = "skin capacity exceeded";
        return {};
    }
    if (source.skinCount != 0 && source.skins == nullptr)
    {
        error = "deformation asset has missing skin storage";
        return {};
    }
    if (source.meshCount == 0 || source.meshCount > limits.maxMeshes)
    {
        error = "deformed mesh capacity exceeded";
        return {};
    }
    if (source.meshes == nullptr)
    {
        error = "deformation asset has missing mesh storage";
        return {};
    }

    try
    {
        std::unordered_set<std::uint64_t> skinIds;
        skinIds.reserve(source.skinCount);
        std::size_t totalJoints = 0;
        std::size_t totalVertices = 0;
        std::size_t totalTargets = 0;
        std::size_t totalValues = 0;
        for (std::size_t index = 0; index < source.skinCount; ++index)
        {
            if (! validateSkin(source.skins[index], limits, totalJoints, totalValues, error))
                return {};
            if (! skinIds.emplace(source.skins[index].id.value).second)
            {
                error = "deformation asset has a duplicate skin identity";
                return {};
            }
        }

        std::unordered_set<std::uint64_t> meshIds;
        meshIds.reserve(source.meshCount);
        for (std::size_t index = 0; index < source.meshCount; ++index)
        {
            if (! validateMesh(source.meshes[index], source, limits, totalVertices,
                               totalTargets, totalValues, error))
                return {};
            if (! meshIds.emplace(source.meshes[index].id.value).second)
            {
                error = "deformation asset has a duplicate mesh identity";
                return {};
            }
        }

        std::shared_ptr<DeformationAsset> result(new DeformationAsset());
        result->skins_.reserve(source.skinCount);
        for (std::size_t skinIndex = 0; skinIndex < source.skinCount; ++skinIndex)
        {
            const auto& sourceSkin = source.skins[skinIndex];
            Skin skin;
            skin.id_ = sourceSkin.id;
            skin.joints_.reserve(sourceSkin.jointCount);
            for (std::size_t jointIndex = 0; jointIndex < sourceSkin.jointCount; ++jointIndex)
            {
                Joint joint;
                joint.id_ = sourceSkin.joints[jointIndex].id;
                joint.parent_ = sourceSkin.joints[jointIndex].parent;
                skin.joints_.push_back(joint);
            }
            skin.inverseBindMatrixValues_.assign(
                sourceSkin.inverseBindMatrixValues,
                sourceSkin.inverseBindMatrixValues + sourceSkin.inverseBindValueCount);
            result->skins_.push_back(std::move(skin));
        }

        result->meshes_.reserve(source.meshCount);
        for (std::size_t meshIndex = 0; meshIndex < source.meshCount; ++meshIndex)
        {
            const auto& sourceMesh = source.meshes[meshIndex];
            Mesh mesh;
            mesh.id_ = sourceMesh.id;
            mesh.vertexCount_ = sourceMesh.vertexCount;
            mesh.skin_ = sourceMesh.skin;
            mesh.jointWeightSets_.reserve(sourceMesh.jointWeightSetCount);
            for (std::size_t setIndex = 0; setIndex < sourceMesh.jointWeightSetCount; ++setIndex)
            {
                const auto& sourceSet = sourceMesh.jointWeightSets[setIndex];
                JointWeightSet set;
                set.jointIndices_.assign(sourceSet.jointIndices,
                                         sourceSet.jointIndices + sourceSet.jointIndexCount);
                set.weights_.assign(sourceSet.weights,
                                    sourceSet.weights + sourceSet.weightCount);
                mesh.jointWeightSets_.push_back(std::move(set));
            }
            for (std::size_t vertex = 0; vertex < sourceMesh.vertexCount; ++vertex)
            {
                double sum = 0.0;
                for (const auto& set : mesh.jointWeightSets_)
                    for (std::size_t component = 0; component < 4; ++component)
                        sum += set.weights_[vertex * 4 + component];
                for (auto& set : mesh.jointWeightSets_)
                    for (std::size_t component = 0; component < 4; ++component)
                        set.weights_[vertex * 4 + component] = static_cast<float>(
                            static_cast<double>(set.weights_[vertex * 4 + component]) / sum);
            }

            mesh.morphTargets_.reserve(sourceMesh.morphTargetCount);
            for (std::size_t targetIndex = 0; targetIndex < sourceMesh.morphTargetCount;
                 ++targetIndex)
            {
                const auto& sourceTarget = sourceMesh.morphTargets[targetIndex];
                MorphTarget target;
                target.id_ = sourceTarget.id;
                target.positionDeltas_.assign(sourceTarget.positionDeltas,
                                              sourceTarget.positionDeltas
                                                  + sourceTarget.positionDeltaCount);
                if (sourceTarget.normalDeltaCount != 0)
                    target.normalDeltas_.assign(sourceTarget.normalDeltas,
                                                sourceTarget.normalDeltas
                                                    + sourceTarget.normalDeltaCount);
                if (sourceTarget.tangentDeltaCount != 0)
                    target.tangentDeltas_.assign(sourceTarget.tangentDeltas,
                                                 sourceTarget.tangentDeltas
                                                     + sourceTarget.tangentDeltaCount);
                mesh.morphTargets_.push_back(std::move(target));
            }
            result->meshes_.push_back(std::move(mesh));
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        error = "deformation asset allocation failed within admitted limits";
        return {};
    }
}

const Skin* DeformationAsset::findSkin (SkinId id) const noexcept
{
    const auto found = std::find_if(skins_.begin(), skins_.end(), [id] (const Skin& skin)
    {
        return skin.id() == id;
    });
    return found == skins_.end() ? nullptr : &*found;
}

const Mesh* DeformationAsset::findMesh (MeshId id) const noexcept
{
    const auto found = std::find_if(meshes_.begin(), meshes_.end(), [id] (const Mesh& mesh)
    {
        return mesh.id() == id;
    });
    return found == meshes_.end() ? nullptr : &*found;
}

std::shared_ptr<const MorphWeightSample> MorphWeightSample::create (
    const DeformationAsset& asset, const MorphWeightSampleView& source,
    const Limits& requestedLimits, std::string& error)
{
    error.clear();
    const auto limits = boundedLimits(requestedLimits);
    if (! source.mesh.isValid())
    {
        error = "morph weight sample has an invalid stable mesh identity";
        return {};
    }
    const auto* mesh = asset.findMesh(source.mesh);
    if (mesh == nullptr)
    {
        error = "morph weight sample references an unknown mesh";
        return {};
    }
    const auto expectedValues = mesh->morphTargets().size();
    if (expectedValues == 0)
    {
        error = "morph weight sample references a mesh without morph targets";
        return {};
    }
    if (source.targetCount != expectedValues || source.valueCount != expectedValues)
    {
        error = "sampled morph target and value cardinalities do not match the mesh";
        return {};
    }
    if (expectedValues > limits.maxMorphTargetsPerMesh
        || expectedValues > limits.maxTotalAccessorValues)
    {
        error = "sampled morph weight capacity exceeded";
        return {};
    }
    if (source.targets == nullptr || source.values == nullptr)
    {
        error = "morph weight sample has missing target or value storage";
        return {};
    }
    for (std::size_t index = 0; index < expectedValues; ++index)
    {
        if (source.targets[index] != mesh->morphTargets()[index].id())
        {
            error = "sampled morph target identities do not match mesh target order";
            return {};
        }
        if (! std::isfinite(source.values[index])
            || std::abs(source.values[index]) > limits.maxMorphWeightMagnitude)
        {
            error = "sampled morph weight is nonfinite or out of range";
            return {};
        }
    }

    try
    {
        std::shared_ptr<MorphWeightSample> result(new MorphWeightSample());
        result->mesh_ = source.mesh;
        result->targets_.assign(source.targets, source.targets + source.targetCount);
        result->values_.assign(source.values, source.values + source.valueCount);
        return result;
    }
    catch (const std::bad_alloc&)
    {
        error = "morph weight sample allocation failed within admitted limits";
        return {};
    }
}
} // namespace visualdeformation
