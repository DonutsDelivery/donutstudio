#pragma once

#include <cstdint>

namespace videohelper::viewportgeneration {
// The live viewport's project generation. The evaluation context and every
// Geometry Core owner must read this same value. A viewport that has not
// bumped its timeline yet uses 1, so no owner is refused as zero, and the
// first timeline bump is also 1.
inline constexpr std::uint64_t projectGeneration(std::uint64_t timelineGeneration) noexcept
{
    return timelineGeneration == 0 ? std::uint64_t{1} : timelineGeneration;
}
} // namespace videohelper::viewportgeneration
