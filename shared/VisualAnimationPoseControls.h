#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace visualanimation
{
inline constexpr std::uint32_t kUnresolvedMorphTargetIndex = 65535;
// A pose may adjust every key of the selected mesh. Morph target indices are
// 16-bit in the pose wire (65535 marks an unresolved key), so this is the index
// range, not a count cap.
inline constexpr std::size_t kMaximumMorphWeightDeltas = 65536;
// Runtime parameter that carries one key's additive weight delta.
inline std::string morphWeightDeltaParameter(std::size_t key)
{
    return "morph" + std::to_string(key) + "Weight";
}
// Parses a canonical "morph<i>Weight" name whose key is a valid morph index.
inline bool parseMorphWeightDeltaParameter(std::string_view name, std::size_t& key) noexcept
{
    constexpr std::string_view head = "morph";
    constexpr std::string_view tail = "Weight";
    if (name.size() <= head.size() + tail.size() || name.substr(0, head.size()) != head
        || name.substr(name.size() - tail.size()) != tail)
        return false;
    const auto digits = name.substr(head.size(), name.size() - head.size() - tail.size());
    if (digits.empty() || digits.size() > 5 || (digits.size() > 1 && digits.front() == '0')) return false;
    std::size_t value = 0;
    for (const char digit : digits)
    {
        if (digit < '0' || digit > '9') return false;
        value = value * 10 + static_cast<std::size_t>(digit - '0');
    }
    if (value >= kMaximumMorphWeightDeltas) return false;
    key = value;
    return true;
}
// One selected object's authored adjustments, applied after its baked animation.
// Bone identity is the existing one-based glTF node ID. Morph index is local
// to the selected mesh and is resolved from its named authoring selector.
// Older saved selectors carry only the catalog ID; admission resolves their index.
struct PoseControls
{
    std::uint64_t meshStableId = 0;
    std::uint64_t boneStableId = 0;
    std::uint32_t morphTargetStableId = 0;
    std::uint32_t morphTargetIndex = 0;
    bool boneEnabled = false;
    bool morphEnabled = false;
    std::array<double, 3> translation {};
    std::array<double, 3> rotationDegrees {};
    std::array<double, 3> scale { 1.0, 1.0, 1.0 };
    double morphWeight = 0.0;
    // Exact one-based glTF node. Zero retains the legacy single-mesh selection.
    std::uint64_t nodeStableId = 0;
    // Additive weight deltas of individual keys of the selected mesh. Each key is its
    // local morph index, as morphTargetIndex is. A delta is added to the key's evaluated
    // weight, and the sum is clamped to [-16, 16]. Keys are unique and empty means none.
    std::vector<std::pair<std::uint32_t, double>> morphWeightDeltas;
};

inline bool validPoseControls(const PoseControls& controls) noexcept
{
    if (controls.nodeStableId > 65536
        || (controls.nodeStableId != 0 && (controls.meshStableId == 0 || controls.meshStableId > 65536)))
        return false;
    if ((controls.boneEnabled || controls.morphEnabled)
        && (controls.meshStableId == 0 || controls.meshStableId > 65536))
        return false;
    if (controls.boneEnabled && (controls.boneStableId == 0
                                 || controls.boneStableId > 65536))
        return false;
    if (controls.morphEnabled && (controls.morphTargetStableId == 0
                                  || controls.morphTargetStableId > 65536))
        return false;
    if (controls.morphTargetIndex >= 65536 || !std::isfinite(controls.morphWeight)
        || std::abs(controls.morphWeight) > 16.0)
        return false;
    if (controls.morphWeightDeltas.size() > kMaximumMorphWeightDeltas
        || (!controls.morphWeightDeltas.empty()
            && (controls.meshStableId == 0 || controls.meshStableId > 65536)))
        return false;
    for (std::size_t entry = 0; entry < controls.morphWeightDeltas.size(); ++entry)
    {
        const auto& [key, weight] = controls.morphWeightDeltas[entry];
        if (key >= 65536 || !std::isfinite(weight) || std::abs(weight) > 16.0)
            return false;
        for (std::size_t earlier = 0; earlier < entry; ++earlier)
            if (controls.morphWeightDeltas[earlier].first == key)
                return false;
    }
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!std::isfinite(controls.translation[axis])
            || std::abs(controls.translation[axis]) > 10000.0
            || !std::isfinite(controls.rotationDegrees[axis])
            || std::abs(controls.rotationDegrees[axis]) > 3600.0
            || !std::isfinite(controls.scale[axis])
            || controls.scale[axis] < static_cast<double>(0.0001f) || controls.scale[axis] > 1000.0)
            return false;
    return true;
}
} // namespace visualanimation
