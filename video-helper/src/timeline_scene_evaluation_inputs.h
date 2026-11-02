#pragma once
#include "canonical_block_c_frame.h"

#include <array>
#include <cstdint>
#include <cstring>

namespace videohelper::geometry {
using TimelineSceneEvaluationInputs = std::array<std::uint64_t, 17>;
// Everything a timeline mesh reads besides its authored scene and plan. Water replay reads the
// score and beat map, and the frame key names both by generation.
inline TimelineSceneEvaluationInputs timelineSceneEvaluationInputs(double seconds, double projectSeconds,
    bool hasWaterHistory, const canonicalblockc::FrameKey* frame) noexcept {
  const auto bits = [](double value) {
    std::uint64_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
  };
  TimelineSceneEvaluationInputs inputs {};
  inputs[0] = bits(seconds);
  inputs[1] = bits(projectSeconds);
  inputs[2] = (hasWaterHistory ? 1u : 0u) | (frame ? 2u : 0u);
  if (frame) {
    const auto& key = *frame;
    inputs[3] = key.carrierVersion; inputs[4] = key.projectGeneration; inputs[5] = key.sourceGeneration;
    inputs[6] = key.helperGeneration; inputs[7] = key.backendGeneration; inputs[8] = key.deviceGeneration;
    inputs[9] = key.scoreGeneration; inputs[10] = key.beatMapGeneration; inputs[11] = key.fpsGeneration;
    inputs[12] = key.loopGeneration; inputs[13] = key.seekGeneration;
    inputs[14] = static_cast<std::uint64_t>(key.frame); inputs[15] = bits(key.beat); inputs[16] = bits(key.fps);
  }
  return inputs;
}
} // namespace videohelper::geometry
