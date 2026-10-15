#include "VisualDeformationContract.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace visualdeformation;

int failures = 0;

void check (bool condition, const char* message)
{
    if (condition)
        return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

bool near (float actual, float expected, float tolerance = 1.0e-5f)
{
    return std::abs(actual - expected) <= tolerance;
}

std::array<float, 16> identityMatrix()
{
    return { 1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f };
}

struct Fixture
{
    std::vector<JointView> joints { { JointId { 101 }, {} },
                                    { JointId { 102 }, JointId { 101 } } };
    std::vector<float> inverseBindValues;
    std::vector<std::uint32_t> indices0 { 0, 1, 0, 0, 1, 0, 0, 0 };
    std::vector<float> weights0 { 0.25f, 0.5f, 0.0f, 0.0f,
                                  1.0f, 0.0f, 0.0f, 0.0f };
    std::vector<std::uint32_t> indices1 { 0, 0, 0, 0, 0, 0, 0, 0 };
    std::vector<float> weights1 { 0.25f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f };
    std::vector<float> targetAPositions { 2.0f, 0.0f, 0.0f,
                                          0.0f, 2.0f, 0.0f };
    std::vector<float> targetANormals { 0.0f, 1.0f, 0.0f,
                                        0.0f, 1.0f, 0.0f };
    std::vector<float> targetBPositions { 0.0f, 0.0f, 1.0f,
                                          0.0f, 0.0f, 2.0f };
    std::array<JointWeightSetView, 2> sets;
    std::array<MorphTargetView, 2> targets;
    SkinView skin;
    MeshView mesh;
    AssetView asset;

    Fixture()
    {
        const auto identity = identityMatrix();
        inverseBindValues.insert(inverseBindValues.end(), identity.begin(), identity.end());
        inverseBindValues.insert(inverseBindValues.end(), identity.begin(), identity.end());
        refresh();
    }

    void refresh()
    {
        sets = { JointWeightSetView { indices0.data(), indices0.size(),
                                      weights0.data(), weights0.size() },
                 JointWeightSetView { indices1.data(), indices1.size(),
                                      weights1.data(), weights1.size() } };
        targets = { MorphTargetView { MorphTargetId { 301 },
                                      targetAPositions.data(), targetAPositions.size(),
                                      targetANormals.data(), targetANormals.size(), nullptr, 0 },
                    MorphTargetView { MorphTargetId { 302 },
                                      targetBPositions.data(), targetBPositions.size(),
                                      nullptr, 0, nullptr, 0 } };
        skin = { SkinId { 201 }, joints.data(), joints.size(),
                 inverseBindValues.data(), inverseBindValues.size() };
        mesh = { MeshId { 401 }, 2, skin.id, sets.data(), sets.size(),
                 targets.data(), targets.size() };
        asset = { &skin, 1, &mesh, 1 };
    }
};

std::shared_ptr<const DeformationAsset> admit (Fixture& fixture, std::string& error,
                                                Limits limits = {})
{
    fixture.refresh();
    return DeformationAsset::create(fixture.asset, limits, error);
}

bool rejected (Fixture& fixture, std::string* diagnostic = nullptr, Limits limits = {})
{
    std::string error;
    fixture.refresh();
    const auto result = DeformationAsset::create(fixture.asset, limits, error);
    if (diagnostic != nullptr)
        *diagnostic = error;
    return result == nullptr && ! error.empty();
}

// Test oracle only. Production code receives the admitted data and performs the
// same operations in an admitted GPU backend. This deliberately small routine
// must never become a renderer fallback.
using Matrix = std::array<float, 16>;
using Position = std::array<float, 3>;

Matrix multiply (const Matrix& left, const float* right)
{
    Matrix result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row]
                    += left[inner * 4 + row] * right[column * 4 + inner];
    return result;
}

Position transformPoint (const Matrix& matrix, const Position& point)
{
    return {
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14]
    };
}

std::vector<Position> referenceDeform (const Mesh& mesh, const Skin& skin,
                                       const MorphWeightSample& sample,
                                       const std::vector<Position>& basePositions,
                                       const std::vector<Matrix>& jointWorldMatrices)
{
    std::vector<Matrix> palettes;
    palettes.reserve(skin.joints().size());
    for (std::size_t joint = 0; joint < skin.joints().size(); ++joint)
        palettes.push_back(multiply(jointWorldMatrices[joint],
                                    skin.inverseBindMatrixValues().data() + joint * 16));

    auto positions = basePositions;
    for (std::size_t target = 0; target < mesh.morphTargets().size(); ++target)
        for (std::size_t vertex = 0; vertex < positions.size(); ++vertex)
            for (std::size_t component = 0; component < 3; ++component)
                positions[vertex][component]
                    += sample.values()[target]
                     * mesh.morphTargets()[target].positionDeltas()[vertex * 3 + component];

    std::vector<Position> result(positions.size(), { 0.0f, 0.0f, 0.0f });
    for (std::size_t vertex = 0; vertex < positions.size(); ++vertex)
        for (const auto& set : mesh.jointWeightSets())
            for (std::size_t component = 0; component < 4; ++component)
            {
                const auto offset = vertex * 4 + component;
                const auto weight = set.weights()[offset];
                const auto transformed = transformPoint(palettes[set.jointIndices()[offset]],
                                                        positions[vertex]);
                for (std::size_t axis = 0; axis < 3; ++axis)
                    result[vertex][axis] += weight * transformed[axis];
            }
    return result;
}
} // namespace

int main()
{
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<const DeformationAsset&>().meshes())>>,
        "admitted meshes must be exposed through an immutable collection");
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<const Mesh&>().jointWeightSets())>>,
        "admitted skin weights must be exposed through an immutable collection");
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<const MorphWeightSample&>().values())>>,
        "sampled morph weights must be exposed through an immutable collection");

    std::string error;
    Fixture valid;
    auto asset = admit(valid, error);
    check(asset != nullptr && error.empty(), "valid skin and morph contracts are admitted");
    check(asset && asset->skins().size() == 1 && asset->meshes().size() == 1,
          "admission retains exact skin and mesh cardinalities");
    check(asset && asset->skins()[0].id() == SkinId { 201 }
          && asset->skins()[0].joints()[1].id() == JointId { 102 }
          && asset->skins()[0].joints()[1].parent() == JointId { 101 }
          && asset->meshes()[0].id() == MeshId { 401 }
          && asset->meshes()[0].skin() == SkinId { 201 }
          && asset->meshes()[0].morphTargets()[1].id() == MorphTargetId { 302 },
          "stable skin, joint, mesh, binding, and target identities survive admission");
    check(asset && asset->findSkin(SkinId { 201 }) == &asset->skins()[0]
          && asset->findMesh(MeshId { 401 }) == &asset->meshes()[0],
          "stable identities resolve within the immutable asset");
    check(asset && near(asset->meshes()[0].jointWeightSets()[0].weights()[0], 0.25f)
          && near(asset->meshes()[0].jointWeightSets()[1].weights()[0], 0.25f),
          "weights spanning multiple sets remain normalized per vertex");

    if (asset)
    {
        valid.inverseBindValues[0] = 9.0f;
        valid.weights0[0] = 0.0f;
        valid.targetAPositions[0] = 99.0f;
        check(asset->skins()[0].inverseBindMatrixValues()[0] == 1.0f
              && asset->meshes()[0].jointWeightSets()[0].weights()[0] == 0.25f
              && asset->meshes()[0].morphTargets()[0].positionDeltas()[0] == 2.0f,
              "admission owns immutable copies of matrices, weights, and deltas");

        const std::array<MorphTargetId, 2> targetIds {
            MorphTargetId { 301 }, MorphTargetId { 302 }
        };
        std::array<float, 2> sampleValues { 0.5f, -0.25f };
        const MorphWeightSampleView sampleView { MeshId { 401 }, targetIds.data(),
                                                 targetIds.size(), sampleValues.data(),
                                                 sampleValues.size() };
        auto sample = MorphWeightSample::create(*asset, sampleView, {}, error);
        check(sample != nullptr && error.empty(),
              "finite bounded morph weights matching stable target order are admitted");
        sampleValues[0] = 7.0f;
        check(sample && sample->values()[0] == 0.5f,
              "sample admission owns an immutable weight copy");

        if (sample)
        {
            Matrix joint0 = identityMatrix();
            Matrix joint1 = identityMatrix();
            joint0[12] = 10.0f;
            joint1[13] = 20.0f;
            const std::vector<Position> basePositions { { 0.0f, 0.0f, 0.0f },
                                                        { 1.0f, 0.0f, 0.0f } };
            const auto first = referenceDeform(asset->meshes()[0], asset->skins()[0],
                                               *sample, basePositions, { joint0, joint1 });
            const auto repeated = referenceDeform(asset->meshes()[0], asset->skins()[0],
                                                  *sample, basePositions, { joint0, joint1 });
            check(first == repeated, "test-only reference deformation is deterministic");
            check(first.size() == 2
                  && near(first[0][0], 6.0f) && near(first[0][1], 10.0f)
                  && near(first[0][2], -0.25f)
                  && near(first[1][0], 1.0f) && near(first[1][1], 21.0f)
                  && near(first[1][2], -0.5f),
                  "test oracle applies morph deltas before weighted joint palettes");
        }

        {
            auto wrongTargets = targetIds;
            std::swap(wrongTargets[0], wrongTargets[1]);
            const MorphWeightSampleView wrong { MeshId { 401 }, wrongTargets.data(), 2,
                                                sampleValues.data(), 2 };
            check(! MorphWeightSample::create(*asset, wrong, {}, error)
                  && error == "sampled morph target identities do not match mesh target order",
                  "sampled weights cannot drift onto a different stable target order");
        }
        {
            const MorphWeightSampleView shortSample { MeshId { 401 }, targetIds.data(), 2,
                                                      sampleValues.data(), 1 };
            check(! MorphWeightSample::create(*asset, shortSample, {}, error)
                  && error == "sampled morph target and value cardinalities do not match the mesh",
                  "sampled morph weight accessor cardinality is exact");
        }
        {
            auto badValues = sampleValues;
            badValues[0] = std::numeric_limits<float>::quiet_NaN();
            const MorphWeightSampleView nonfinite { MeshId { 401 }, targetIds.data(), 2,
                                                    badValues.data(), 2 };
            check(! MorphWeightSample::create(*asset, nonfinite, {}, error)
                  && error == "sampled morph weight is nonfinite or out of range",
                  "nonfinite sampled morph weights are rejected");
        }
        {
            std::array<float, 2> excessive { 17.0f, 0.0f };
            Limits relaxed;
            relaxed.maxMorphWeightMagnitude = 1000.0f;
            const MorphWeightSampleView outOfRange { MeshId { 401 }, targetIds.data(), 2,
                                                     excessive.data(), 2 };
            check(! MorphWeightSample::create(*asset, outOfRange, relaxed, error)
                  && error == "sampled morph weight is nonfinite or out of range",
                  "callers cannot relax the hard sampled-weight bound");
        }
    }

    {
        Fixture fixture;
        fixture.joints[0].parent = JointId { 102 };
        fixture.joints[1].parent = JointId { 101 };
        check(rejected(fixture, &error)
              && error == "skin joint hierarchy contains a cycle",
              "cyclic joint hierarchies are rejected");
    }
    {
        Fixture fixture;
        fixture.joints[1].parent = JointId { 999 };
        check(rejected(fixture, &error)
              && error == "skin joint parent is outside the skin",
              "joint parents must resolve inside their skin");
    }
    {
        Fixture fixture;
        fixture.joints[1].id = fixture.joints[0].id;
        check(rejected(fixture, &error) && error == "skin has a duplicate joint identity",
              "duplicate stable joint identities are rejected");
    }
    {
        Fixture fixture;
        fixture.skin.inverseBindValueCount = 31;
        fixture.asset = { &fixture.skin, 1, &fixture.mesh, 1 };
        const auto result = DeformationAsset::create(fixture.asset, {}, error);
        check(! result
              && error == "inverse bind matrix accessor cardinality does not match the joint count",
              "inverse bind accessor cardinality is exactly sixteen scalars per joint");
    }
    {
        Fixture fixture;
        fixture.inverseBindValues[7] = std::numeric_limits<float>::infinity();
        check(rejected(fixture, &error)
              && error == "inverse bind matrix contains a nonfinite or out-of-range value",
              "nonfinite inverse bind values are rejected");
    }
    {
        Fixture fixture;
        fixture.indices0.pop_back();
        check(rejected(fixture, &error)
              && error == "joint and weight accessor cardinalities do not match the vertex count",
              "joint accessor cardinality is exactly four indices per vertex and set");
    }
    {
        Fixture fixture;
        fixture.weights1.pop_back();
        check(rejected(fixture, &error)
              && error == "joint and weight accessor cardinalities do not match the vertex count",
              "weight accessor cardinality is exactly four weights per vertex and set");
    }
    {
        Fixture fixture;
        fixture.indices0[0] = 2;
        check(rejected(fixture, &error)
              && error == "joint accessor index is outside the bound skin",
              "joint indices are bounded by the referenced skin");
    }
    {
        Fixture fixture;
        fixture.weights0[0] = -0.25f;
        check(rejected(fixture, &error) && error == "skin weight is negative or nonfinite",
              "negative skin weights are rejected");
    }
    {
        Fixture fixture;
        fixture.weights0[0] = std::numeric_limits<float>::quiet_NaN();
        check(rejected(fixture, &error) && error == "skin weight is negative or nonfinite",
              "nonfinite skin weights are rejected");
    }
    {
        Fixture fixture;
        fixture.weights1[0] = 0.0f;
        check(rejected(fixture, &error)
              && error == "skin weights for a vertex are not normalized",
              "skin influence sets must sum to one per vertex");
    }
    {
        Fixture fixture;
        fixture.targetAPositions.pop_back();
        check(rejected(fixture, &error)
              && error == "morph target accessor cardinality does not match the vertex count",
              "morph position accessor cardinality is exactly three deltas per vertex");
    }
    {
        Fixture fixture;
        fixture.targetANormals.pop_back();
        check(rejected(fixture, &error)
              && error == "morph target accessor cardinality does not match the vertex count",
              "optional normal accessor cardinality is exact when present");
    }
    {
        Fixture fixture;
        fixture.targetBPositions[0] = std::numeric_limits<float>::infinity();
        check(rejected(fixture, &error)
              && error == "morph target contains a missing, nonfinite, or out-of-range delta",
              "nonfinite morph deltas are rejected");
    }
    {
        Fixture fixture;
        fixture.targets[1].id = fixture.targets[0].id;
        fixture.mesh.morphTargets = fixture.targets.data();
        fixture.asset = { &fixture.skin, 1, &fixture.mesh, 1 };
        const auto result = DeformationAsset::create(fixture.asset, {}, error);
        check(! result && error == "mesh has a duplicate morph target identity",
              "duplicate stable morph target identities are rejected");
    }
    {
        Fixture fixture;
        fixture.mesh.skin = SkinId { 999 };
        fixture.asset = { &fixture.skin, 1, &fixture.mesh, 1 };
        const auto result = DeformationAsset::create(fixture.asset, {}, error);
        check(! result
              && error == "mesh-to-skin binding references an unknown stable skin identity",
              "mesh-to-skin bindings require an admitted stable skin identity");
    }
    {
        Fixture fixture;
        Limits limits;
        limits.maxVerticesPerMesh = 1;
        check(rejected(fixture, &error, limits)
              && error == "deformed mesh vertex capacity exceeded",
              "per-mesh vertex limits apply before accessor copies");
    }
    {
        Fixture fixture;
        Limits limits;
        limits.maxJointsPerSkin = 1;
        check(rejected(fixture, &error, limits) && error == "skin joint capacity exceeded",
              "per-skin joint limits apply before hierarchy copies");
    }
    {
        Fixture fixture;
        Limits limits;
        limits.maxMorphTargetsPerMesh = 1;
        check(rejected(fixture, &error, limits)
              && error == "mesh morph target capacity exceeded",
              "per-mesh morph target limits apply before delta copies");
    }
    {
        Fixture fixture;
        Limits limits;
        limits.maxTotalAccessorValues = 31;
        check(rejected(fixture, &error, limits)
              && error == "deformation accessor value capacity exceeded",
              "total accessor scalar values are bounded before immutable copies");
    }

    std::fprintf(stderr, failures == 0
        ? "Skin and morph deformation contract checks passed\n"
        : "%d skin and morph deformation contract checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
