#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace visualdeformation
{
template <typename Tag>
struct StableId
{
    std::uint64_t value = 0;

    constexpr bool isValid() const noexcept { return value != 0; }
    constexpr bool operator== (StableId other) const noexcept { return value == other.value; }
    constexpr bool operator!= (StableId other) const noexcept { return value != other.value; }
    constexpr bool operator< (StableId other) const noexcept { return value < other.value; }
};

struct SkinIdTag {};
struct JointIdTag {};
struct MeshIdTag {};
struct MorphTargetIdTag {};

using SkinId = StableId<SkinIdTag>;
using JointId = StableId<JointIdTag>;
using MeshId = StableId<MeshIdTag>;
using MorphTargetId = StableId<MorphTargetIdTag>;

struct Limits
{
    std::size_t maxSkins = 256;
    std::size_t maxMeshes = 4096;
    std::size_t maxJointsPerSkin = 1024;
    std::size_t maxTotalJoints = 65536;
    std::size_t maxVerticesPerMesh = 4u * 1024u * 1024u;
    std::size_t maxTotalVertices = 8u * 1024u * 1024u;
    std::size_t maxInfluenceSetsPerMesh = 4;
    std::size_t maxMorphTargetsPerMesh = 256;
    std::size_t maxTotalMorphTargets = 8192;
    std::size_t maxTotalAccessorValues = 256u * 1024u * 1024u;
    float weightSumTolerance = 1.0e-4f;
    float maxInverseBindElementMagnitude = 1.0e8f;
    float maxMorphDeltaMagnitude = 1.0e6f;
    float maxMorphWeightMagnitude = 16.0f;
};

struct JointView
{
    JointId id;
    JointId parent;
};

// Inverse bind matrices are packed as column-major 4x4 matrices. Their scalar
// count must be exactly jointCount * 16.
struct SkinView
{
    SkinId id;
    const JointView* joints = nullptr;
    std::size_t jointCount = 0;
    const float* inverseBindMatrixValues = nullptr;
    std::size_t inverseBindValueCount = 0;
};

// Each set is a glTF-style VEC4 JOINTS_n/WEIGHTS_n pair. Both scalar counts
// must be exactly mesh.vertexCount * 4. Weights across every set for one vertex
// form one normalized influence vector.
struct JointWeightSetView
{
    const std::uint32_t* jointIndices = nullptr;
    std::size_t jointIndexCount = 0;
    const float* weights = nullptr;
    std::size_t weightCount = 0;
};

// Position deltas are required. Normal and tangent deltas are optional, but an
// included accessor must contain exactly vertexCount * 3 scalar values.
struct MorphTargetView
{
    MorphTargetId id;
    const float* positionDeltas = nullptr;
    std::size_t positionDeltaCount = 0;
    const float* normalDeltas = nullptr;
    std::size_t normalDeltaCount = 0;
    const float* tangentDeltas = nullptr;
    std::size_t tangentDeltaCount = 0;
};

struct MeshView
{
    MeshId id;
    std::size_t vertexCount = 0;
    SkinId skin;
    const JointWeightSetView* jointWeightSets = nullptr;
    std::size_t jointWeightSetCount = 0;
    const MorphTargetView* morphTargets = nullptr;
    std::size_t morphTargetCount = 0;
};

struct AssetView
{
    const SkinView* skins = nullptr;
    std::size_t skinCount = 0;
    const MeshView* meshes = nullptr;
    std::size_t meshCount = 0;
};

class Joint
{
public:
    JointId id() const noexcept { return id_; }
    JointId parent() const noexcept { return parent_; }

private:
    friend class DeformationAsset;
    JointId id_;
    JointId parent_;
};

class Skin
{
public:
    SkinId id() const noexcept { return id_; }
    const std::vector<Joint>& joints() const noexcept { return joints_; }
    const std::vector<float>& inverseBindMatrixValues() const noexcept
    {
        return inverseBindMatrixValues_;
    }

private:
    friend class DeformationAsset;
    SkinId id_;
    std::vector<Joint> joints_;
    std::vector<float> inverseBindMatrixValues_;
};

class JointWeightSet
{
public:
    const std::vector<std::uint32_t>& jointIndices() const noexcept { return jointIndices_; }
    const std::vector<float>& weights() const noexcept { return weights_; }

private:
    friend class DeformationAsset;
    std::vector<std::uint32_t> jointIndices_;
    std::vector<float> weights_;
};

class MorphTarget
{
public:
    MorphTargetId id() const noexcept { return id_; }
    const std::vector<float>& positionDeltas() const noexcept { return positionDeltas_; }
    const std::vector<float>& normalDeltas() const noexcept { return normalDeltas_; }
    const std::vector<float>& tangentDeltas() const noexcept { return tangentDeltas_; }
    bool hasNormalDeltas() const noexcept { return ! normalDeltas_.empty(); }
    bool hasTangentDeltas() const noexcept { return ! tangentDeltas_.empty(); }

private:
    friend class DeformationAsset;
    MorphTargetId id_;
    std::vector<float> positionDeltas_;
    std::vector<float> normalDeltas_;
    std::vector<float> tangentDeltas_;
};

class Mesh
{
public:
    MeshId id() const noexcept { return id_; }
    std::size_t vertexCount() const noexcept { return vertexCount_; }
    SkinId skin() const noexcept { return skin_; }
    const std::vector<JointWeightSet>& jointWeightSets() const noexcept
    {
        return jointWeightSets_;
    }
    const std::vector<MorphTarget>& morphTargets() const noexcept { return morphTargets_; }

private:
    friend class DeformationAsset;
    MeshId id_;
    std::size_t vertexCount_ = 0;
    SkinId skin_;
    std::vector<JointWeightSet> jointWeightSets_;
    std::vector<MorphTarget> morphTargets_;
};

// This type only owns validated inputs for a GPU backend. CPU deformation math
// is intentionally absent; deterministic reference math belongs in tests only.
class DeformationAsset
{
public:
    static std::shared_ptr<const DeformationAsset> create (const AssetView& source,
                                                            const Limits& limits,
                                                            std::string& error);

    const std::vector<Skin>& skins() const noexcept { return skins_; }
    const std::vector<Mesh>& meshes() const noexcept { return meshes_; }
    const Skin* findSkin (SkinId id) const noexcept;
    const Mesh* findMesh (MeshId id) const noexcept;

private:
    std::vector<Skin> skins_;
    std::vector<Mesh> meshes_;
};

struct MorphWeightSampleView
{
    MeshId mesh;
    const MorphTargetId* targets = nullptr;
    std::size_t targetCount = 0;
    const float* values = nullptr;
    std::size_t valueCount = 0;
};

class MorphWeightSample
{
public:
    static std::shared_ptr<const MorphWeightSample> create (
        const DeformationAsset& asset, const MorphWeightSampleView& source,
        const Limits& limits, std::string& error);

    MeshId mesh() const noexcept { return mesh_; }
    const std::vector<MorphTargetId>& targets() const noexcept { return targets_; }
    const std::vector<float>& values() const noexcept { return values_; }

private:
    MeshId mesh_;
    std::vector<MorphTargetId> targets_;
    std::vector<float> values_;
};
} // namespace visualdeformation
