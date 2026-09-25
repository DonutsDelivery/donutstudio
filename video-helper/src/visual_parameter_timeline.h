#pragma once
#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace videorender
{
struct VisualParameterSample { std::string paramId; double atSec=0, value=0; };
class VisualParameterTimeline
{
public:
    bool bind(const std::vector<VisualParameterSample>& samples, std::string& error)
    {
        tracks_.clear();
        if (samples.size()>262144) { error="Simulation automation exceeds 262144 samples"; return false; }
        for (const auto& sample:samples)
        {
            if (sample.paramId.empty() || sample.paramId.size()>256
                || !std::isfinite(sample.atSec) || sample.atSec<0 || !std::isfinite(sample.value))
            { error="Malformed simulation automation sample"; return false; }
            auto& track=tracks_[sample.paramId];
            if (!track.empty() && sample.atSec<=track.back().first)
            { error="Simulation automation sample times must increase strictly per destination"; return false; }
            track.emplace_back(sample.atSec,sample.value);
        }
        if (tracks_.size()>128) { error="Simulation automation exceeds 128 destinations"; return false; }
        return true;
    }
    void sample(const std::string& destination,double seconds,double& value) const
    {
        const auto found=tracks_.find(destination);
        if (found==tracks_.end()) return;
        const auto& track=found->second;
        // These are exact 120 Hz authoring evaluations, held until the next
        // fixed step. Do not interpolate reset edges or AutomationClip steps.
        const auto next=std::upper_bound(track.begin(),track.end(),seconds+1.0e-9,
            [](double time,const auto& point) { return time<point.first; });
        value=(next==track.begin() ? track.front():*std::prev(next)).second;
    }
private:
    std::map<std::string,std::vector<std::pair<double,double>>> tracks_;
};
}
