#pragma once

#include "beat_timeline.h"
#include "video_control_plan.h"
#include <functional>
#include <map>
#include <memory>
#include <set>

namespace videorender
{
// Replays the existing route/control evaluators on their project value grid.
// The solver samples these immutable columns at its rational fixed-step times;
// neither live UI values nor previous seek order participate in evaluation.
class VisualParameterHistory
{
public:
    using AudioReader = std::function<bool(double, const std::vector<int>&, bool,
                                           arbitmod::Audio&, std::string&)>;
    bool bind(const std::vector<arbitmod::Routing>& routes, const videocontrol::Plan& plan,
              std::shared_ptr<const arbitmod::Score> score, videotime::BeatTimeline timeline,
              double fps, const std::set<std::string>& destinations, std::string& error)
    {
        VisualParameterHistory candidate;
        if (!candidate.bindSnapshot(routes,plan,std::move(score),std::move(timeline),fps,destinations,error))
            return false;
        *this=std::move(candidate);
        return true;
    }
private:
    bool bindSnapshot(const std::vector<arbitmod::Routing>& routes, const videocontrol::Plan& plan,
              std::shared_ptr<const arbitmod::Score> score, videotime::BeatTimeline timeline,
              double fps, const std::set<std::string>& destinations, std::string& error)
    {
        if (!score || !std::isfinite(fps) || fps <= 0 || fps > 480 || routes.size() > 128)
        { error = "Visual parameter history has invalid score, rate or route capacity"; return false; }
        routes_.clear();
        for (const auto& route:routes)
            if (route.enabled && destinations.count(route.destination)) routes_.push_back(route);
        // Retain only dependencies of the requested sinks. Unrelated live-only
        // sources must not block an otherwise replayable simulation.
        auto selected=plan;
        selected.operations.clear();
        std::set<int> needed;
        for (auto it=plan.operations.rbegin();it!=plan.operations.rend();++it)
        {
            const bool retain=(it->kind=="sink" && it->enabled && destinations.count(it->destination))
                || std::any_of(it->outputSlots.begin(),it->outputSlots.end(),
                    [&](int slot) { return needed.count(slot)!=0; });
            if (!retain) continue;
            selected.operations.push_back(*it);
            for (const auto& input:it->inputs) needed.insert(input.begin(),input.end());
        }
        std::reverse(selected.operations.begin(),selected.operations.end());
        // No retained Signal operation means this subplan has no Signal window.
        if (std::none_of(selected.operations.begin(),selected.operations.end(),
            [](const auto& op) { return op.kind=="signal.window"; }))
        { selected.version=1; selected.signalWindow={}; }
        if (!executor_.bind(selected,error)) return false;
        needsAudio_=false; needsMasterAudio_=false; audioSources_.clear(); bandSources_.clear();
        const auto inspect=[&](const arbitmod::ModSource& source)
        {
            const bool audio=source.type==arbitmod::SourceType::AudioRms
                || source.type==arbitmod::SourceType::AudioPeak || source.type==arbitmod::SourceType::AudioOnset
                || source.type==arbitmod::SourceType::AudioBand;
            needsAudio_=needsAudio_ || audio;
            if (!audio) return true;
            if (source.type==arbitmod::SourceType::AudioBand && (source.band<0 || source.band>=64))
                return false;
            if (source.trackId<0) needsMasterAudio_=true;
            else
            {
                audioSources_.push_back(source.trackId);
                if (source.type==arbitmod::SourceType::AudioBand)
                    bandSources_.emplace_back(source.trackId,source.band);
            }
            return true;
        };
        for (const auto& route:routes_)
            if (!inspect(route.source)) { error="Simulation history requires an AudioBand index in the canonical 64-band range"; return false; }
        for (const auto& op:selected.operations)
            if (op.kind=="source" && !inspect(op.source))
            { error="Simulation history requires an AudioBand index in the canonical 64-band range"; return false; }
        std::sort(audioSources_.begin(),audioSources_.end());
        audioSources_.erase(std::unique(audioSources_.begin(),audioSources_.end()),audioSources_.end());
        if (audioSources_.size()>8)
        { error="Simulation history exceeds eight selected tracks or groups"; return false; }
        score_=std::move(score); timeline_=std::move(timeline); fps_=fps;
        states_.assign(routes_.size(),{}); columns_.clear(); rows_.clear();
        for (const auto& route : routes_)
            if (route.enabled) columns_.push_back({route.destination,0,route.mode,true});
        for (const auto& operation : selected.operations)
            if (operation.kind=="sink" && operation.enabled)
                columns_.push_back({operation.destination,0,operation.mode,true});
        if (columns_.size()>128)
        { error="Visual parameter history exceeds 128 routed columns"; return false; }
        return true;
    }
public:
    bool sample(const std::string& destination, double seconds, double& value,
                const AudioReader& audioAt, std::string& error)
    {
        if (columns_.empty() || std::none_of(columns_.begin(),columns_.end(),
            [&](const auto& column) { return column.destination==destination; })) return true;
        if (!std::isfinite(seconds) || seconds<0 || seconds*fps_>216000)
        { error="Visual parameter history exceeds its 216000-frame replay bound"; return false; }
        const auto frame=static_cast<std::size_t>(std::floor(seconds*fps_+1.0e-9));
        if ((frame+1)*columns_.size()>4194304)
        { error="Visual parameter history exceeds 4194304 retained values"; return false; }
        while (rows_.size()<=frame)
        {
            const auto index=rows_.size();
            const double time=double(index)/fps_;
            const auto clock=timeline_.clockAtSeconds(time);
            const float beats=static_cast<float>(timeline_.secondsToBeat(time+1/fps_)-clock.beat);
            arbitmod::Clock input {static_cast<float>(clock.beat),static_cast<float>(clock.bpm),
                                  static_cast<float>(clock.beatsPerBar)};
            arbitmod::Audio audio;
            if (needsAudio_ && (!audioAt || !audioAt(time,audioSources_,needsMasterAudio_,audio,error)))
            { if (error.empty()) error="Simulation modulation requires baked project audio history"; return false; }
            if (audio.sourceCount!=static_cast<int>(audioSources_.size()))
            { error="Simulation history requires exactly the selected source identities"; return false; }
            for (int i=0;i<audio.sourceCount;++i)
                for (int j=0;j<i;++j)
                    if (audio.sources[static_cast<std::size_t>(i)].trackId==audio.sources[static_cast<std::size_t>(j)].trackId)
                    { error="Simulation history has duplicate selected source identities"; return false; }
            for (const auto trackId:audioSources_)
            {
                const auto* source=audio.sourceFor(trackId);
                if (!source || (source->tapPoint!=1 && source->tapPoint!=2))
                { error="Simulation history is missing a selected post-FX/pre-fader source"; return false; }
            }
            for (const auto& [trackId,band]:bandSources_)
            {
                const auto* source=audio.sourceFor(trackId);
                if (!source || !source->bandsReady
                    || !std::isfinite(source->bands[static_cast<std::size_t>(band)]))
                { error="Simulation history is missing a finite selected band on the project analysis grid"; return false; }
            }
            if (!executor_.signalWindowAvailable(static_cast<std::int64_t>(index),fps_))
            { error="Visual parameter history is outside the admitted Signal window"; return false; }
            std::vector<float> row;
            row.reserve(columns_.size());
            for (std::size_t i=0;i<routes_.size();++i)
                if (routes_[i].enabled)
                {
                    arbitmod::evaluateRouting(routes_[i],states_[i],0,*score_,input,audio,beats);
                    row.push_back(states_[i].smoothed);
                }
            const auto& sinks=executor_.evaluate(*score_,input,audio,beats,
                static_cast<float>(1/fps_),static_cast<std::int64_t>(index),fps_);
            for (const auto& sink:sinks) if (sink.enabled) row.push_back(sink.value);
            if (row.size()!=columns_.size() || std::any_of(row.begin(),row.end(),
                [](float v) { return !std::isfinite(v); }))
            { error="Visual parameter history has an unavailable or nonfinite routed value"; return false; }
            rows_.push_back(std::move(row));
        }
        for (std::size_t i=0;i<columns_.size();++i)
            if (columns_[i].destination==destination)
                value=arbitmod::combine(columns_[i].mode,static_cast<float>(value),rows_[frame][i]);
        return true;
    }
private:
    bool needsAudio_=false, needsMasterAudio_=false;
    std::vector<int> audioSources_;
    std::vector<std::pair<int,int>> bandSources_;
    double fps_=0;
    std::vector<arbitmod::Routing> routes_;
    std::vector<arbitmod::RoutingState> states_;
    videocontrol::Executor executor_;
    std::shared_ptr<const arbitmod::Score> score_;
    videotime::BeatTimeline timeline_;
    std::vector<videocontrol::SinkValue> columns_;
    std::vector<std::vector<float>> rows_;
};
}
