// block_b_audio_features.h: one Block B analyzer frame mapped to visual AudioFeatures.
//
// The live viewport (ring drain and stopped baked preview) and the exporter's
// audioFeatureAt all turn an arbitblockb::FeatureFrame into the visual uniform
// block. This header is the single place that mapping lives, so rms, peak,
// onset, onsetAge and bands cannot drift between those paths.
//
// Header-only, no GL, no state. Depends only on the analyzer and the visual
// AudioFeatures struct (shader_generator.h, which is GL-free above its
// ARBIT_HAVE_VIEWPORT guard).

#pragma once

#include "block_b_analyzer.h"
#include "shader_generator.h" // videorender::AudioFeatures

#include <algorithm>
#include <vector>

namespace videorender
{

// Overwrites every field of `out` from frame `f`. `out.bands` keeps its capacity
// when `out` is reused across frames, which the live drain relies on. A caller
// may set `out.onset` afterward. The live path does so through mergeDrainOnset.
inline void fillAudioFeatures (AudioFeatures& out, const arbitblockb::FeatureFrame& f)
{
    out.rms = f.rms;
    out.peak = f.peak;
    out.onset = f.onset;
    out.onsetAge = f.onsetAge;
    out.bands.assign (f.bands.begin(), f.bands.end());
}

// Sets `dst.onset` for one live drain that produced `frames` (the frames of this
// tick, in emission order). The onset is a decaying envelope, so the newest frame
// can sit well below a trigger that fired on an earlier hop of the same drain.
// The drain's onset is the maximum envelope across its frames. A trigger (1.0)
// on any hop survives, and the tail still falls with the envelope, because the
// value only drops frame by frame until the next trigger. The previous value of
// `dst.onset` is ignored. An empty `frames` sets 0.
inline void mergeDrainOnset (AudioFeatures& dst, const std::vector<arbitblockb::FeatureFrame>& frames)
{
    float envelope = 0.0f;
    for (const arbitblockb::FeatureFrame& f : frames)
        envelope = std::max (envelope, f.onset);
    dst.onset = envelope;
}

} // namespace videorender
