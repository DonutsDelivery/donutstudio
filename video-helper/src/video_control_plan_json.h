#pragma once

#include "video_control_plan.h"
#include <nlohmann/json.hpp>

namespace videocontrol
{
inline arbitmod::SourceType parseSourceType (const std::string& s)
{
    using ST = arbitmod::SourceType;
    if (s == "NotePitch")       return ST::NotePitch;
    if (s == "NoteVelocity")    return ST::NoteVelocity;
    if (s == "NoteGate")        return ST::NoteGate;
    if (s == "NoteTrigger")     return ST::NoteTrigger;
    if (s == "NoteCount")       return ST::NoteCount;
    if (s == "NoteAge")         return ST::NoteAge;
    if (s == "CentsFromRoot")   return ST::CentsFromRoot;
    if (s == "PrimeEnergy")     return ST::PrimeEnergy;
    if (s == "RootTrigger")     return ST::RootTrigger;
    if (s == "ClockBeatPhase")  return ST::ClockBeatPhase;
    if (s == "ClockBarPhase")   return ST::ClockBarPhase;
    if (s == "ClockBeat")       return ST::ClockBeat;
    if (s == "HarmRatio")       return ST::HarmRatio;
    if (s == "HarmRatioLog2")   return ST::HarmRatioLog2;
    if (s == "HarmRatioNum")    return ST::HarmRatioNum;
    if (s == "HarmRatioDen")    return ST::HarmRatioDen;
    if (s == "HarmLissajous")   return ST::HarmLissajous;
    if (s == "HarmBeatingRate") return ST::HarmBeatingRate;
    if (s == "HarmTenney")      return ST::HarmTenney;
    if (s == "HarmLinkRatio")   return ST::HarmLinkRatio;
    if (s == "Env")             return ST::Env;
    if (s == "Lfo")             return ST::Lfo;
    if (s == "AudioRms")        return ST::AudioRms;
    if (s == "AudioPeak")       return ST::AudioPeak;
    if (s == "AudioOnset")      return ST::AudioOnset;
    if (s == "AudioBand")       return ST::AudioBand;
    return ST::ClockBeatPhase;
}
inline arbitmod::Curve parseCurve (const std::string& s)
{
    if (s == "Exp")    return arbitmod::Curve::Exp;
    if (s == "Log")    return arbitmod::Curve::Log;
    if (s == "SCurve") return arbitmod::Curve::SCurve;
    return arbitmod::Curve::Linear;
}
inline arbitmod::Mode parseMode (const std::string& s)
{
    if (s == "Multiply") return arbitmod::Mode::Multiply;
    if (s == "Replace")  return arbitmod::Mode::Replace;
    return arbitmod::Mode::Add;
}
inline bool parsePlanJson(const nlohmann::json& value, videocontrol::Plan& destination,
                                      std::string& error)
{
    try
    {
        Plan plan;
        if (value.is_null()) { destination = {}; return true; }
        if (!value.is_object())
        {
            error = "controlPlan must be an object";
            return false;
        }
        plan.version = value.value("version", 1);
        plan.numSlots = value.value("numSlots", 0);
        if (value.contains("signalWindow"))
        {
            const auto& window = value["signalWindow"];
            if (!window.is_object() || window.value("policy", std::string{}) != "floor-frame-v1")
            {
                error = "unsupported video Signal window policy";
                return false;
            }
            plan.signalWindow.sampleRate = window.value("sampleRate", 0);
            plan.signalWindow.fps = window.value("fps", 0.0);
            plan.signalWindow.frameCount = window.value("frameCount", 0);
            if (!plan.signalWindow.valid() || value.dump().size() > kMaxSignalPlanBytes)
            {
                error = "video Signal window identity or 2 MiB plan capacity is invalid";
                return false;
            }
        }
        if (!value.contains("operations") || !value["operations"].is_array())
        {
            error = "controlPlan.operations must be an array";
            return false;
        }
        if (value["operations"].size() > videocontrol::kMaxOperations)
        {
            error = "video control plan operation count exceeds capacity";
            return false;
        }

        std::size_t totalFrameValues = 0;
        for (const auto& item : value["operations"])
        {
            if (!item.is_object())
            {
                error = "video control plan operation must be an object";
                return false;
            }
            videocontrol::Operation operation;
            operation.nodeId = item.value("nodeId", -1);
            operation.kind = item.value("kind", std::string{});
            operation.destination = item.value("destination", std::string{});
            operation.targetClipId = item.value("targetClipId", -1);
            operation.targetNodeId = item.value("targetNodeId", -1);
            operation.targetParamId = item.value("targetParamId", std::string{});
            operation.depth = item.value("depth", 1.0f);
            operation.curve = parseCurve(item.value("curve", std::string("Linear")));
            operation.smoothingBeats = item.value("smoothing", 0.0f);
            operation.mode = parseMode(item.value("mode", std::string("Add")));
            operation.enabled = item.value("enabled", true);

            if (item.contains("frameValues"))
            {
                const auto& values = item["frameValues"];
                totalFrameValues += values.size();
                if (!values.is_array() || totalFrameValues > kMaxSignalFrameValues)
                {
                    error = "video Signal frame-value capacity or array is invalid";
                    return false;
                }
                for (const auto& number : values)
                {
                    if (!number.is_number())
                    {
                        error = "video Signal frame value must be numeric";
                        return false;
                    }
                    operation.frameValues.push_back(number.get<float>());
                }
            }

            if (item.contains("inputs") && item["inputs"].is_array())
                for (const auto& input : item["inputs"])
                {
                    if (!input.is_array())
                    {
                        error = "video control plan input must be an array";
                        return false;
                    }
                    std::vector<int> fanIn;
                    for (const auto& slot : input)
                        if (slot.is_number_integer()) fanIn.push_back(slot.get<int>());
                        else
                        {
                            error = "video control plan input slot must be an integer";
                            return false;
                        }
                    operation.inputs.push_back(std::move(fanIn));
                }

            if (item.contains("params") && item["params"].is_array())
                for (const auto& parameter : item["params"])
                    if (parameter.is_number()) operation.params.push_back(parameter.get<float>());
                    else
                    {
                        error = "video control plan parameter must be numeric";
                        return false;
                    }

            if (item.contains("outputSlots") && item["outputSlots"].is_array())
                for (const auto& slot : item["outputSlots"])
                    if (slot.is_number_integer()) operation.outputSlots.push_back(slot.get<int>());
                    else
                    {
                        error = "video control plan output slot must be an integer";
                        return false;
                    }

            if (operation.kind == "source")
            {
                operation.source.type = parseSourceType(
                    item.value("sourceType", std::string("ClockBeatPhase")));
                const auto parameter = [&operation](std::size_t index, float fallback)
                {
                    return index < operation.params.size() ? operation.params[index] : fallback;
                };
                operation.source.trackId = static_cast<int>(std::lround(parameter(0, -1.0f)));
                operation.source.pitchLo = parameter(1, 0.0f);
                operation.source.pitchHi = parameter(2, 127.0f);
                operation.source.primeIndex = static_cast<int>(std::lround(parameter(3, 1.0f)));
                operation.source.axis = static_cast<int>(std::lround(parameter(4, 0.0f)));
                operation.source.linkId = static_cast<int>(std::lround(parameter(5, 0.0f)));
                operation.source.band = static_cast<int>(std::lround(parameter(6, 0.0f)));
                operation.source.lissajousK = static_cast<int>(std::lround(parameter(7, 7.0f)));
                operation.source.triggerDecayBeats = parameter(8, 0.5f);
                operation.source.adsr = { parameter(9, 0.05f), parameter(10, 0.1f),
                                          parameter(11, 0.8f), parameter(12, 0.2f),
                                          parameter(13, 0.0f) };
                operation.source.lfo.periodBeats = parameter(14, 1.0f);
                operation.source.lfo.phase0 = parameter(15, 0.0f);
                operation.source.lfo.seed = static_cast<uint32_t>(
                    std::lround(parameter(16, 1.0f)));
                operation.source.lfo.shape = static_cast<arbitmod::LFOShape>(std::clamp(
                    static_cast<int>(std::lround(parameter(17, 0.0f))), 0, 4));
                operation.source.lfo.hz = parameter(18, 0.0f) >= 0.5f;
                operation.source.lfo.rateHz = parameter(19, 1.0f);
                operation.source.lfo.retrigger = parameter(20, 0.0f) >= 0.5f;
            }
            plan.operations.push_back(std::move(operation));
        }
        if (!videocontrol::validatePlan(plan, error)) return false;
        destination = std::move(plan);
        return true;
    }
    catch (const nlohmann::json::exception&)
    {
        error = "invalid video control plan field type or range";
        return false;
    }
}

} // namespace videocontrol
