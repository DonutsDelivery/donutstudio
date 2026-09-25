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
#include <memory>
#include "../../shared/GeometrySpectrumField.h"

#include "block_b_analyzer.h"   // arbitblockb::FeatureFrame

namespace videohelper
{
struct AnalysisSourcePath { int trackId = -1; bool group = false; std::string path; };
struct AnalysisSourceStream {
    int trackId = -1;
    bool group = false;
    double sampleRate = 0;
    std::vector<arbitblockb::FeatureFrame> frames;
};
using AnalysisSources = std::vector<AnalysisSourceStream>;
inline videowire::geometry::spectrum::SourceFeaturesAt sourceSpectrumReader(
    std::shared_ptr<const AnalysisSources> sources)
{
    return [sources=std::move(sources)](videowire::geometry::spectrum::Source source,
                                      std::int32_t trackId, double seconds,
                                      videowire::geometry::spectrum::Bands& bands)
    {
        bands={};
        if (!sources || !std::isfinite(seconds) || seconds<0) return false;
        for (const auto& stream : *sources)
            if (stream.trackId==trackId
                && stream.group==(source==videowire::geometry::spectrum::Source::group)
                && source!=videowire::geometry::spectrum::Source::master
                && stream.sampleRate>0 && !stream.frames.empty())
            {
                const double sample=seconds*stream.sampleRate/arbitblockb::kHop;
                if (sample>=static_cast<double>(stream.frames.size()+1)) return true;
                const auto index=std::max<std::int64_t>(0,static_cast<std::int64_t>(std::floor(sample))-1);
                bands=stream.frames[static_cast<size_t>(index)].bands;
                return true;
            }
        return false;
    };
}
std::shared_ptr<const AnalysisSources> analyzeSourceMixes(
    const std::vector<AnalysisSourcePath>& paths, std::string& error);

// Per-hop Block B feature frames for the WHOLE WAV (empty on any failure).
// srOut = decoded sample rate (0 on failure). Byte-identical to the exporter's
// in-line analysis of the same file ⇒ a stopped preview frame at time t reads
// the same feature the export frame at t does (see exporter.cpp audioFeatureAt
// and viewport.cpp previewMixFeatureAt — both use idx = floor(t*sr/kHop)-1).
std::vector<arbitblockb::FeatureFrame> analyzeMixWavOffline (const std::string& wavPath,
                                                             double& srOut);
} // namespace videohelper
