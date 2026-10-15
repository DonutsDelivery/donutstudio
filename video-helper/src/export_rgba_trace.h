#pragma once

#include <cmath>
#include <cstdint>
#include <string>

namespace videohelper::exporttrace
{
// This diagnostic is limited to the Plan 3 beat-one comparison at 30 fps.
inline std::string validate(int frameIndex, const std::string& path,
                            double fps, double rangeStartSec, int64_t totalFrames)
{
    if (frameIndex < 0 && path.empty()) return {};
    if (frameIndex != 15 || path.empty()) return "invalid export RGBA trace selection";
    if (!std::isfinite(fps) || std::abs(fps - 30.0) > 1e-9
        || std::abs(rangeStartSec) > 1e-9 || totalFrames <= frameIndex)
        return "export RGBA trace requires frame 15 at timeline 0.5 seconds";
    return {};
}

inline bool selected(int requestedIndex, int64_t encodedPts, bool readbackOwned)
{
    return readbackOwned && requestedIndex >= 0 && encodedPts == requestedIndex;
}
} // namespace videohelper::exporttrace
