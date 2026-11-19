#pragma once

#include "VisualVolumeData.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace visualvolume
{
inline constexpr const char* sourceKind = "visual.volume.density";
inline constexpr const char* renderKind = "visual.volume.render";
inline constexpr std::uint32_t maximumResolution = 64;
inline constexpr std::size_t maximumVoxels = 64u * 64u * 64u;
inline constexpr std::string_view prefix = "volume-density-v1:";

// A bounded density field, not a framebuffer or filesystem capability. The
// source's authored controls are frozen into the render operation. Native GL
// and Metal consume the same admitted voxel data and raymarch it on the GPU.
struct Operation final
{
    std::uint32_t sourceStableId = 0;
    std::uint32_t renderStableId = 0;
    std::uint32_t resolution = 32;
    float radius = 0.38f;
    float density = 0.65f;
    float noiseAmount = 0.35f;
    float noiseFrequency = 4.0f;
    float phase = 0.0f;
};

struct Parameter final
{
    const char* id;
    const char* label;
    float minimum, maximum, initial;
    bool integral;
};

inline constexpr std::array<Parameter, 6> parameters {{
    {"resolution", "Resolution", 8.0f, 64.0f, 32.0f, true},
    {"radius", "Radius", 0.02f, 0.75f, 0.38f, false},
    {"density", "Density", 0.0f, 1.0f, 0.65f, false},
    {"noiseAmount", "Noise Amount", 0.0f, 1.0f, 0.35f, false},
    {"noiseFrequency", "Noise Frequency", 0.0f, 16.0f, 4.0f, false},
    {"phase", "Phase", -100.0f, 100.0f, 0.0f, false}
}};

inline bool valid(const Operation& value) noexcept
{
    const auto bounded = [](float x, float low, float high)
        { return std::isfinite(x) && x >= low && x <= high; };
    return value.sourceStableId > 0 && value.renderStableId > 0
        && value.sourceStableId != value.renderStableId
        && value.resolution >= 8 && value.resolution <= maximumResolution
        && bounded(value.radius, 0.02f, 0.75f) && bounded(value.density, 0.0f, 1.0f)
        && bounded(value.noiseAmount, 0.0f, 1.0f)
        && bounded(value.noiseFrequency, 0.0f, 16.0f)
        && bounded(value.phase, -100.0f, 100.0f);
}

inline std::string encode(const Operation& value)
{
    if (!valid(value)) return {};
    static_assert(sizeof(float) == sizeof(std::uint32_t)
        && std::numeric_limits<float>::is_iec559, "volume wire requires IEEE-754 binary32");
    std::array<std::uint32_t, 8> words {value.sourceStableId, value.renderStableId, value.resolution};
    const std::array<float, 5> controls {value.radius, value.density, value.noiseAmount,
                                       value.noiseFrequency, value.phase};
    for (std::size_t i = 0; i < controls.size(); ++i)
    {
        const float canonical = controls[i] == 0.0f ? 0.0f : controls[i];
        std::memcpy(&words[i + 3], &canonical, sizeof(canonical));
    }
    std::string wire(prefix);
    constexpr char digits[] = "0123456789abcdef";
    for (const auto word : words)
        for (int shift = 28; shift >= 0; shift -= 4)
            wire.push_back(digits[(word >> shift) & 15u]);
    return wire;
}

inline bool decode(std::string_view wire, Operation& value)
{
    if (wire.size() != prefix.size() + 64 || wire.substr(0, prefix.size()) != prefix) return false;
    std::array<std::uint32_t, 8> words {};
    for (std::size_t i = 0; i < 64; ++i)
    {
        const char c = wire[prefix.size() + i];
        const int digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (digit < 0) return false;
        words[i / 8] = (words[i / 8] << 4u) | static_cast<std::uint32_t>(digit);
    }
    Operation candidate;
    candidate.sourceStableId = words[0]; candidate.renderStableId = words[1]; candidate.resolution = words[2];
    std::array<float*, 5> controls {&candidate.radius, &candidate.density, &candidate.noiseAmount,
                                  &candidate.noiseFrequency, &candidate.phase};
    for (std::size_t i = 0; i < controls.size(); ++i) std::memcpy(controls[i], &words[i + 3], sizeof(float));
    if (!valid(candidate) || encode(candidate) != wire) return false;
    value = candidate;
    return true;
}

inline bool makeDensity(const Operation& operation, videowire::DenseVolumeDescriptor& destination)
{
    if (!valid(operation)) return false;
    videowire::DenseVolumeDescriptor value;
    value.dimensions = {operation.resolution, operation.resolution, operation.resolution};
    value.format = videowire::VolumeVoxelFormat::densityU8;
    const auto size = static_cast<std::size_t>(operation.resolution);
    value.voxels.resize(size * size * size);
    constexpr double tau = 6.28318530717958647692;
    for (std::size_t z = 0; z < size; ++z)
        for (std::size_t y = 0; y < size; ++y)
            for (std::size_t x = 0; x < size; ++x)
            {
                const double px = (static_cast<double>(x) + 0.5) / size - 0.5;
                const double py = (static_cast<double>(y) + 0.5) / size - 0.5;
                const double pz = (static_cast<double>(z) + 0.5) / size - 0.5;
                const double r = std::sqrt(px * px + py * py + pz * pz);
                const double edge = std::clamp((operation.radius - r) * size / 2.0, 0.0, 1.0);
                const double fade = edge * edge * (3.0 - 2.0 * edge);
                const double wave = 0.5 + 0.5 * std::sin(tau * operation.noiseFrequency * px + operation.phase)
                    * std::sin(tau * operation.noiseFrequency * py + 0.7 * operation.phase)
                    * std::sin(tau * operation.noiseFrequency * pz - 0.4 * operation.phase);
                const double density = operation.density * fade
                    * ((1.0 - operation.noiseAmount) + operation.noiseAmount * wave);
                value.voxels[(z * size + y) * size + x] = static_cast<std::uint8_t>(
                    std::lround(std::clamp(density, 0.0, 1.0) * 255.0));
            }
    destination = std::move(value);
    return true;
}
} // namespace visualvolume
