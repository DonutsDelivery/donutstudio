#pragma once

#include "../../../shared/RenderPassCompositeContract.h"
#include <array>
#include <cmath>

namespace renderpassfixture
{
inline renderpasscomposite::Program hdrBranches()
{
    using namespace renderpasscomposite;
    Program result;
    result.count = 4;
    result.output = 3;
    auto& first = result.steps[0].parameters;
    first.mode = Mode::DepthFog;
    first.farDepth = 0.001f;
    first.fogColor = {4.0f, 0.25f, 0.5f};
    first.exposure = 1;
    // This display setting must never run on the linear branch connection.
    first.outputTransform = OutputTransform::ReinhardToSRGB;
    auto& second = result.steps[1].parameters;
    second.mode = Mode::DepthFog;
    second.farDepth = 0.001f;
    second.fogColor = {0.5f, 0.5f, 0.5f};
    second.fogColorSpace = ColorInput::SRGB;
    result.steps[2].parameters.mode = Mode::Add;
    result.steps[2].parameters.amount = 0.5f;
    result.steps[2].inputA = 0;
    result.steps[2].inputB = 1;
    result.steps[3].parameters.mode = Mode::Mix;
    result.steps[3].parameters.amount = 0.5f;
    result.steps[3].parameters.exposure = -1;
    result.steps[3].parameters.outputTransform = OutputTransform::ReinhardToSRGB;
    result.steps[3].inputA = 2;
    result.steps[3].inputB = 0;
    return result;
}

inline std::array<int, 3> expectedSdr()
{
    const double fogLinear = std::pow((0.5 + 0.055) / 1.055, 2.4);
    std::array<int, 3> result {};
    const std::array<double, 3> first {4.0, 0.25, 0.5};
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        const double hdr = first[i] + fogLinear * 0.125;
        const double mapped = hdr / (1.0 + hdr);
        result[i] = static_cast<int>(std::lround(255.0 * (1.055 * std::pow(mapped, 1.0 / 2.4) - 0.055)));
    }
    return result;
}
} // namespace renderpassfixture
