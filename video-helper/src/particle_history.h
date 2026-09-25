#pragma once

#include "particle_parameters.h"
#include "beat_timeline.h"
#include "block_b_analyzer.h"
#include "../../shared/VisualAnimationDeformationEvaluation.h"
#include "../../shared/ParticleGeometryContract.h"
#include "../../shared/VisualSimulationAutomation.h"

namespace videorender
{
// Owned score and tempo snapshots plus the frame's read-only project analysis.
// The audio reader is consumed synchronously while the source frame is alive.
struct ParticleHistoryAudio
{
    float rms = 0, onset = 0;
    std::int64_t onsetSample = -1;
};

struct ParticleHistorySource
{
    std::shared_ptr<const arbitmod::Score> score;
    videotime::BeatTimeline timeline;
    canonicalblockc::FrameKey key;
    const std::vector<arbitblockb::FeatureFrame>* audioFrames = nullptr;
    double audioSampleRate = 0;
    double parameterFrameSeconds = 1.0/30;
    std::function<bool(const std::string&, double, double&, std::string&)> parameterAt;
    std::function<bool(videowire::geometry::RetainedMeshData&,
        const visualdeformation::RationalFrameTime&, std::uint64_t, std::string&)> importedGeometry;
    bool hasAudio() const { return audioFrames && !audioFrames->empty() && audioSampleRate > 0; }
    ParticleHistoryAudio audioAt(double seconds) const;
};

inline constexpr std::int64_t kParticleReplayTicksPerSecond = 28224000;
inline constexpr std::int64_t kParticleReplayStepTicks = 235200;
inline constexpr int kParticleReplayMaxEvents = 512;

inline std::array<float*, visualsimulation::parameters.size()> particleAutomationFields(ParticleParams& p)
{
    return {{&p.gravity,&p.gravityX,&p.gravityZ,&p.drag,&p.angularDrag,&p.restitution,
        &p.impulseX,&p.impulseY,&p.impulseZ,&p.attraction,&p.linkSpring,&p.linkRatioInfluence,
        &p.linkRestScale,&p.bodyMass,&p.friction,&p.bodyRadius,&p.force,&p.rmsGain,&p.onsetGain,
        &p.emissionRate,&p.reset,&p.resetTime}};
}

inline bool sampleParticleParameters(const ParticleParams& authored, ParticleParams& p,
                                     double seconds, std::string& error)
{
    auto base=authored;
    auto fields=particleAutomationFields(p), defaults=particleAutomationFields(base);
    for (std::size_t i=0;i<fields.size();++i)
    {
        const auto& parameter=visualsimulation::parameters[i];
        double value=*defaults[i];
        if (p.history && p.history->parameterAt && p.historyNodeId>0)
            if (!p.history->parameterAt("clip"+std::to_string(p.historyClipId)+"/visual"
                +std::to_string(p.historyNodeId)+"/"+parameter.name,seconds,value,error)) return false;
        if (!std::isfinite(value)) { error="Simulation automation produced a nonfinite value"; return false; }
        *fields[i]=std::clamp(static_cast<float>(value),parameter.minimum,parameter.maximum);
    }
    if (p.simulationSpace==0 && (p.gravityX!=0 || p.gravityZ!=0 || p.impulseZ!=0 || p.angularDrag!=0))
    { error="Automated depth and angular controls require Solid 3D"; return false; }
    if ((p.rmsGain>0 || p.onsetGain>0) && p.history && !p.history->hasAudio())
    { error="Automated audio forces require baked project audio history"; return false; }
    return true;
}

struct ParticleReplayWindow { std::int64_t start = 0, age = 0; };

inline ParticleReplayWindow particleReplayWindow(const ParticleParams& p, double clipSeconds)
{
    const auto elapsed = std::llround((clipSeconds - p.resetTime) * kParticleReplayTicksPerSecond);
    const auto duration = std::llround(double(std::clamp(p.lifetime, 0.1f, 10.0f)) * kParticleReplayTicksPerSecond);
    const auto age = p.resetMode == 0 ? elapsed % duration : std::min(elapsed, duration);
    const auto start = std::llround((p.historyProjectSeconds - clipSeconds + p.resetTime) * kParticleReplayTicksPerSecond)
        + (p.resetMode == 0 ? elapsed - age : 0);
    return {start, age};
}

// Bound work by actual events inside this replay window, not the total number
// of authored notes in the project. Over-cap histories fail before GPU work.
template <typename Visitor>
bool visitParticleHistoryEvents(const ParticleParams& p, ParticleReplayWindow window, Visitor visit)
{
    int count = 0;
    const auto add = [&](std::int64_t tick)
    {
        tick -= window.start;
        if (tick <= 0 || tick >= window.age) return true;
        if (++count > kParticleReplayMaxEvents) return false;
        visit(tick);
        return true;
    };
    if (p.geometryCount == 0)
        for (const auto& note : p.history->score->notes)
            for (const auto beat : {note.startBeat, note.endBeat()})
                if (!add(std::llround(p.history->timeline.beatToSeconds(beat) * kParticleReplayTicksPerSecond)))
                    return false;
    if (p.history->hasAudio())
    {
        const auto& frames = *p.history->audioFrames;
        const double firstSample = double(window.start) / kParticleReplayTicksPerSecond * p.history->audioSampleRate;
        const double finalSample = double(window.start + window.age) / kParticleReplayTicksPerSecond * p.history->audioSampleRate;
        auto next = std::lower_bound(frames.begin(), frames.end(), firstSample,
            [](const auto& value, double sample) { return value.sampleIndex < sample; });
        for (; next != frames.end() && next->sampleIndex <= finalSample; ++next)
            if (next->onset > 0 && next->onsetAge == 0
                && !add(std::llround(double(next->sampleIndex) / p.history->audioSampleRate * kParticleReplayTicksPerSecond)))
                return false;
    }
    return true;
}

inline ParticleHistoryAudio particleHistoryAudioAt(
    const std::vector<arbitblockb::FeatureFrame>& frames, double rate, double seconds)
{
    ParticleHistoryAudio out;
    if (frames.empty() || rate <= 0 || !std::isfinite(seconds) || seconds < 0) return out;
    const auto index = std::clamp<std::int64_t>(
        static_cast<std::int64_t>(std::floor(seconds * rate / arbitblockb::kHop + 1.0e-9)) - 1,
        0, static_cast<std::int64_t>(frames.size()) - 1);
    const auto& frame = frames[static_cast<std::size_t>(index)];
    out.rms = frame.rms;
    out.onset = frame.onset;
    // Block B's hop grid is the event authority. Quantizing its onset age to
    // hops avoids changing a sample identity through float rounding.
    if (frame.onset > 0 && frame.onsetAge < arbitblockb::kOnsetAgeMax)
        out.onsetSample = frame.sampleIndex - static_cast<std::int64_t>(
            std::llround(frame.onsetAge * rate / arbitblockb::kHop)) * arbitblockb::kHop;
    return out;
}

inline ParticleHistoryAudio ParticleHistorySource::audioAt(double seconds) const
{
    return hasAudio() ? particleHistoryAudioAt(*audioFrames, audioSampleRate, seconds) : ParticleHistoryAudio{};
}

inline bool particleHistoryModAudio(const ParticleHistorySource& source, double seconds,
                                   arbitmod::Audio& out, std::string& error)
{
    if (!source.hasAudio())
    { error="Simulation modulation requires baked project audio for playback, seeking and export"; return false; }
    const auto& frames=*source.audioFrames;
    const auto index=std::clamp<std::int64_t>(
        static_cast<std::int64_t>(std::floor(seconds*source.audioSampleRate/arbitblockb::kHop+1.0e-9))-1,
        0,static_cast<std::int64_t>(frames.size())-1);
    const auto& frame=frames[static_cast<std::size_t>(index)];
    out.rms=frame.rms; out.peak=frame.peak; out.onset=frame.onset;
    std::copy_n(frame.bands.begin(),64,out.bands);
    return true;
}

inline bool particleHistoryReady(const ParticleParams& params, std::string& error, double clipSeconds = 0)
{
    if (!params.historicalReplay) return true;
    if (!params.history || !params.history->score)
    { error = "Historical particle replay requires the project score and tempo snapshot"; return false; }
    if ((params.rmsGain > 0 || params.onsetGain > 0) && !params.history->hasAudio())
    { error = "Historical audio-reactive particles require baked project audio for playback, seeking and export"; return false; }
    if (!params.history->score->notes.empty())
    {
        auto key = params.history->key;
        key.frame = 0; key.fps = kParticleReplayTicksPerSecond; key.beat = 0;
        if (!canonicalblockc::FrameProducer::admissible(key, params.history->score, 0))
        { error = "Historical particles require an admitted canonical score"; return false; }
    }
    if (std::isfinite(clipSeconds) && clipSeconds >= params.resetTime
        && !visitParticleHistoryEvents(params, particleReplayWindow(params, clipSeconds), [](std::int64_t) {}))
    { error = "Historical particle replay exceeds 512 note/audio events in its bounded replay window; shorten the lifetime"; return false; }
    return true;
}
}
