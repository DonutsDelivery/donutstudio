#pragma once
// mix_analyze.h — frame-perfect audio parity (Media Machine sweep).
//
// One place to decode a baked master-mix WAV and run the Block B OFFLINE
// analyzer — the SAME path the exporter uses (decodeWavToMonoFloat →
// BlockBAnalyzer::analyzeOffline). The LIVE viewport calls this so that, while
// the transport is STOPPED/scrubbing (no audio flowing through the live ring),
// audio-reactive shaders and mod routings read identical features to export
// instead of zero-feeding. Defined in exporter.cpp (libav + the analyzer
// already live there); only available when the helper is built with the
// viewport (ARBIT_HAVE_VIEWPORT).
#include <string>
#include <vector>

#include "block_b_analyzer.h"   // arbitblockb::FeatureFrame

namespace videohelper
{
// Per-hop Block B feature frames for the WHOLE WAV (empty on any failure).
// srOut = decoded sample rate (0 on failure). Byte-identical to the exporter's
// in-line analysis of the same file ⇒ a stopped preview frame at time t reads
// the same feature the export frame at t does (see exporter.cpp audioFeatureAt
// and viewport.cpp previewMixFeatureAt — both use idx = floor(t*sr/kHop)-1).
std::vector<arbitblockb::FeatureFrame> analyzeMixWavOffline (const std::string& wavPath,
                                                             double& srOut);
} // namespace videohelper
