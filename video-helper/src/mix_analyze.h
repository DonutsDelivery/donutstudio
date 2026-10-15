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

#include "mod_defs.h"
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
// Both consumers use the released source stream and its project-zero hop grid.
inline const arbitblockb::FeatureFrame* sourceFeatureAt(const AnalysisSourceStream& stream,
                                                       double seconds)
{
    const double sample=seconds*stream.sampleRate/arbitblockb::kHop;
    if (stream.frames.empty() || !std::isfinite(sample) || sample < 0
        || sample>=static_cast<double>(stream.frames.size()+1)) return nullptr;
    const auto index=std::max<std::int64_t>(0,static_cast<std::int64_t>(std::floor(sample))-1);
    return &stream.frames[static_cast<std::size_t>(index)];
}
inline bool sourceScalarAudioAt(const AnalysisSources* sources, const std::vector<int>& selected,
                                double seconds, arbitmod::Audio& audio, std::string& error)
{
    auto candidate=audio;
    candidate.sourceCount=0;
    if (!std::isfinite(seconds) || seconds<0 || selected.size()>audio.sources.size()
        || (sources && sources->size()>videowire::geometry::spectrum::kMaximumAnalysisSources))
    { error="Simulation source analysis has invalid time or exceeds eight sources"; return false; }
    for (std::size_t slot=0;slot<selected.size();++slot)
    {
        const auto trackId=selected[slot];
        if (std::find(selected.begin(),selected.begin()+slot,trackId)!=selected.begin()+slot)
        { error="Simulation source analysis requests a duplicate identity"; return false; }
        const AnalysisSourceStream* found=nullptr;
        if (sources) for (const auto& stream:*sources)
            if (stream.trackId==trackId)
            {
                if (found) { error="Simulation source analysis has a duplicate track/group identity"; return false; }
                found=&stream;
            }
        if (trackId<0 || !found || !std::isfinite(found->sampleRate)
            || found->sampleRate<=0 || found->frames.empty())
        { error="Simulation history is missing the selected track/group project analysis"; return false; }
        auto& value=candidate.sources[static_cast<std::size_t>(candidate.sourceCount++)];
        value={trackId,found->group ? 2:1,0,0,0};
        if (const auto* frame=sourceFeatureAt(*found,seconds))
        {
            if (!std::isfinite(frame->rms) || !std::isfinite(frame->peak) || !std::isfinite(frame->onset))
            { error="Simulation source analysis has nonfinite scalar values"; return false; }
            value.rms=frame->rms; value.peak=frame->peak; value.onset=frame->onset;
            const auto index=static_cast<std::size_t>(frame-found->frames.data());
            value.bandsReady=frame->sampleIndex==static_cast<std::int64_t>(index+1)*arbitblockb::kHop;
            value.bands=frame->bands;
        }
    }
    audio=candidate;
    return true;
}
// The analyzer frame index for a timeline time: idx = floor(t*sr/kHop) - 1,
// clamped into the analysis. Export, the stopped preview and every reader below
// call this one function, so a time resolves to the same frame in each path.
// Times before the first window clamp to frame 0. Requires a nonempty analysis
// and a positive sample rate.
inline std::size_t analysisFrameIndex(std::size_t frameCount, double sampleRate, double seconds)
{
    const double targetSample = std::max(0.0, seconds) * sampleRate;
    const auto index = static_cast<std::int64_t>(std::floor(targetSample / arbitblockb::kHop)) - 1;
    return static_cast<std::size_t>(std::clamp<std::int64_t>(index, 0,
        static_cast<std::int64_t>(frameCount) - 1));
}

// Project-analysis readers. A bands reader and a scalars reader over the same
// frames return the rms, peak and onset of the frame that supplies the bands.
// The readers keep a reference, so the caller must keep the frames alive.
inline videowire::geometry::spectrum::FeaturesAt analysisBandsReader(
    const std::vector<arbitblockb::FeatureFrame>& frames, double sampleRate)
{
    return [&frames, sampleRate](double seconds)
    {
        videowire::geometry::spectrum::Bands bands {};
        if (!frames.empty() && sampleRate > 0.0)
            bands = frames[analysisFrameIndex(frames.size(), sampleRate, seconds)].bands;
        return bands;
    };
}
inline videowire::geometry::spectrum::ScalarsAt analysisScalarsReader(
    const std::vector<arbitblockb::FeatureFrame>& frames, double sampleRate)
{
    return [&frames, sampleRate](double seconds, videowire::geometry::spectrum::Scalars& scalars)
    {
        scalars = {};
        if (frames.empty() || sampleRate <= 0.0) return true;
        const auto& frame = frames[analysisFrameIndex(frames.size(), sampleRate, seconds)];
        scalars.rms = frame.rms; scalars.peak = frame.peak; scalars.onset = frame.onset;
        return true;
    };
}

// The selected track or group analysis stream, or null when it is absent.
// Bands and scalars use this same lookup, so they cannot come from different streams.
inline const AnalysisSourceStream* selectedSourceStream(const AnalysisSources* sources,
                                                       videowire::geometry::spectrum::Source source,
                                                       std::int32_t trackId)
{
    if (!sources || source == videowire::geometry::spectrum::Source::master) return nullptr;
    for (const auto& stream : *sources)
        if (stream.trackId == trackId
            && stream.group == (source == videowire::geometry::spectrum::Source::group)
            && stream.sampleRate > 0 && !stream.frames.empty())
            return &stream;
    return nullptr;
}

inline videowire::geometry::spectrum::SourceFeaturesAt sourceSpectrumReader(
    std::shared_ptr<const AnalysisSources> sources)
{
    return [sources=std::move(sources)](videowire::geometry::spectrum::Source source,
                                      std::int32_t trackId, double seconds,
                                      videowire::geometry::spectrum::Bands& bands)
    {
        bands={};
        if (!sources || !std::isfinite(seconds) || seconds<0) return false;
        const auto* stream=selectedSourceStream(sources.get(),source,trackId);
        if (!stream) return false;
        if (const auto* frame=sourceFeatureAt(*stream,seconds)) bands=frame->bands;
        return true;
    };
}
inline videowire::geometry::spectrum::SourceScalarsAt sourceScalarsReader(
    std::shared_ptr<const AnalysisSources> sources)
{
    return [sources=std::move(sources)](videowire::geometry::spectrum::Source source,
                                      std::int32_t trackId, double seconds,
                                      videowire::geometry::spectrum::Scalars& scalars)
    {
        scalars={};
        if (!sources || !std::isfinite(seconds) || seconds<0) return false;
        const auto* stream=selectedSourceStream(sources.get(),source,trackId);
        if (!stream) return false;
        if (const auto* frame=sourceFeatureAt(*stream,seconds))
        {
            scalars.rms=frame->rms; scalars.peak=frame->peak; scalars.onset=frame->onset;
        }
        return true;
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
