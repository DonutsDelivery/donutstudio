#pragma once

#include <string>
#include <vector>

namespace arbitgpu
{
// One GPU submission of an OpenGL grating frame: the span between the timestamps
// at two flushes, named for what it drew.
struct GratingSubmissionSpan
{
    std::string name;
    double milliseconds = 0.0;
};

// The calling thread's last finished OpenGL grating frame, one span per submission
// in order. Empty when the last grating frame was flushed without a finish.
std::vector<GratingSubmissionSpan> lastGratingSubmissions();
} // namespace arbitgpu
