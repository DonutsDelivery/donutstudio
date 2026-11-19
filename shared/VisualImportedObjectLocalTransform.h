#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace visualimportedobjectlocal
{
// GLB node and mesh ids are one-based uint32 values; the parser admits any count.
inline constexpr std::uint32_t kMaximumNodeStableId = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t kMaximumMeshStableId = std::numeric_limits<std::uint32_t>::max();
inline constexpr double kMaximumTranslation = 10000.0;
inline constexpr double kMaximumRotationDegrees = 3600.0;
inline constexpr double kMinimumScale = double(0.0001f);
inline constexpr double kMaximumScale = 1000.0;

// The enclosing imported leaf supplies the exact asset, scene and epochs.
// Mesh zero names a transform-only node; the codec cannot admit its source.
struct ObjectLocalTransformV1 final
{
    std::uint32_t version = 0, nodeStableId = 0, meshStableId = 0;
    std::array<double, 3> translation {};
    std::array<double, 3> rotationDegrees {};
    std::array<double, 3> scale {{ 1.0, 1.0, 1.0 }};
};

static_assert(sizeof(double) == sizeof(std::uint64_t)
    && std::numeric_limits<double>::is_iec559);

inline std::uint64_t bits(double value) noexcept
{
    std::uint64_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline bool same(const ObjectLocalTransformV1& a, const ObjectLocalTransformV1& b) noexcept
{
    if (a.version != b.version || a.nodeStableId != b.nodeStableId
        || a.meshStableId != b.meshStableId) return false;
    for (std::size_t i = 0; i < 3; ++i)
        if (bits(a.translation[i]) != bits(b.translation[i])
            || bits(a.rotationDegrees[i]) != bits(b.rotationDegrees[i])
            || bits(a.scale[i]) != bits(b.scale[i])) return false;
    return true;
}

inline bool valid(const ObjectLocalTransformV1& value) noexcept
{
    if (value.version == 0) return same(value, ObjectLocalTransformV1 {});
    if (value.version != 1 || value.nodeStableId == 0) return false;
    for (std::size_t i = 0; i < 3; ++i)
        if (!std::isfinite(value.translation[i]) || std::abs(value.translation[i]) > kMaximumTranslation
            || !std::isfinite(value.rotationDegrees[i]) || std::abs(value.rotationDegrees[i]) > kMaximumRotationDegrees
            || !std::isfinite(value.scale[i]) || value.scale[i] < kMinimumScale
            || value.scale[i] > kMaximumScale) return false;
    return true;
}
// A neutral transform leaves its node where the source put it.
inline bool neutral(const ObjectLocalTransformV1& value) noexcept
{
    for (std::size_t i = 0; i < 3; ++i)
        if (value.translation[i] != 0 || value.rotationDegrees[i] != 0 || value.scale[i] != 1) return false;
    return true;
}

// Several moved objects of one imported scene. The primary transform is the one
// the inspector's nine fields edit; the others keep their own transforms. A list
// needs a version 1 primary, and every node appears once across both.
inline bool validList(const ObjectLocalTransformV1& primary,
                      const std::vector<ObjectLocalTransformV1>& others) noexcept
{
    if (others.empty()) return true;
    if (primary.version != 1 || !valid(primary)) return false;
    for (std::size_t i = 0; i < others.size(); ++i)
    {
        if (others[i].version != 1 || !valid(others[i])
            || others[i].nodeStableId == primary.nodeStableId) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (others[j].nodeStableId == others[i].nodeStableId) return false;
    }
    return true;
}

inline bool sameList(const std::vector<ObjectLocalTransformV1>& a,
                     const std::vector<ObjectLocalTransformV1>& b) noexcept
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same(a[i], b[i])) return false;
    return true;
}

// The stored transform of a node, or a neutral version 1 transform.
inline ObjectLocalTransformV1 transformFor(const ObjectLocalTransformV1& primary,
    const std::vector<ObjectLocalTransformV1>& others, std::uint32_t node, std::uint32_t mesh) noexcept
{
    if (primary.version == 1 && primary.nodeStableId == node) return primary;
    for (const auto& other : others)
        if (other.nodeStableId == node) return other;
    ObjectLocalTransformV1 fresh;
    fresh.version = 1; fresh.nodeStableId = node; fresh.meshStableId = mesh;
    return fresh;
}

// Makes `value` the primary transform. The old primary joins the others unless it
// is neutral, and a neutral value that was not primary before is still kept as the
// primary so the inspector can edit it.
inline void selectPrimary(ObjectLocalTransformV1& primary, std::vector<ObjectLocalTransformV1>& others,
                          const ObjectLocalTransformV1& value)
{
    std::vector<ObjectLocalTransformV1> kept;
    kept.reserve(others.size() + 1);
    if (primary.version == 1 && primary.nodeStableId != value.nodeStableId && !neutral(primary))
        kept.push_back(primary);
    for (const auto& other : others)
        if (other.nodeStableId != value.nodeStableId && !neutral(other)) kept.push_back(other);
    primary = value;
    others = std::move(kept);
}

// Project storage for the non-primary transforms: entries separated by spaces,
// each "node:mesh:" then nine IEEE doubles as 16 hex digits, translation,
// rotation and scale in order. Exact and locale-free.
inline std::string encodeList(const std::vector<ObjectLocalTransformV1>& values)
{
    std::string text;
    constexpr char hex[] = "0123456789abcdef";
    for (const auto& value : values)
    {
        if (!text.empty()) text.push_back(' ');
        text += std::to_string(value.nodeStableId); text.push_back(':');
        text += std::to_string(value.meshStableId);
        for (const auto* triple : { &value.translation, &value.rotationDegrees, &value.scale })
            for (const auto number : *triple)
            {
                text.push_back(':');
                const auto word = bits(number);
                for (int shift = 60; shift >= 0; shift -= 4) text.push_back(hex[(word >> shift) & 15]);
            }
    }
    return text;
}

inline bool decodeList(std::string_view text, std::vector<ObjectLocalTransformV1>& output)
{
    std::vector<ObjectLocalTransformV1> values;
    const auto decimal = [](std::string_view field, std::uint32_t maximum, std::uint32_t& out) {
        if (field.empty() || field.size() > 5 || (field.size() > 1 && field[0] == '0')) return false;
        std::uint32_t number = 0;
        for (const char c : field) { if (c < '0' || c > '9') return false; number = number * 10 + std::uint32_t(c - '0'); }
        if (number > maximum) return false;
        out = number; return true;
    };
    while (!text.empty())
    {
        const auto end = text.find(' ');
        auto entry = text.substr(0, end);
        text = end == std::string_view::npos ? std::string_view {} : text.substr(end + 1);
        if (end != std::string_view::npos && text.empty()) return false;
        std::string_view fields[11];
        for (std::size_t i = 0; i < 11; ++i)
        {
            const auto colon = entry.find(':');
            if ((i == 10) != (colon == std::string_view::npos)) return false;
            fields[i] = entry.substr(0, colon);
            if (i < 10) entry.remove_prefix(colon + 1);
        }
        ObjectLocalTransformV1 value;
        value.version = 1;
        if (!decimal(fields[0], kMaximumNodeStableId, value.nodeStableId)
            || !decimal(fields[1], kMaximumMeshStableId, value.meshStableId)) return false;
        std::array<double*, 9> numbers {{ &value.translation[0], &value.translation[1], &value.translation[2],
            &value.rotationDegrees[0], &value.rotationDegrees[1], &value.rotationDegrees[2],
            &value.scale[0], &value.scale[1], &value.scale[2] }};
        for (std::size_t i = 0; i < 9; ++i)
        {
            const auto field = fields[i + 2];
            if (field.size() != 16) return false;
            std::uint64_t word = 0;
            for (const char c : field)
            {
                const int nibble = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
                if (nibble < 0) return false;
                word = (word << 4) | std::uint64_t(nibble);
            }
            std::memcpy(numbers[i], &word, sizeof(word));
        }
        values.push_back(value);
    }
    // Each entry is valid and names its own node.
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (!valid(values[i])) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (values[j].nodeStableId == values[i].nodeStableId) return false;
    }
    output = std::move(values);
    return true;
}
} // namespace visualimportedobjectlocal
