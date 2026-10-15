#include "video_control_plan.h"
#include "video_control_plan_json.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

videocontrol::Operation valueOp(int nodeId, float value, int slot)
{
    videocontrol::Operation operation;
    operation.nodeId = nodeId;
    operation.kind = "control.const";
    operation.params = { value };
    operation.outputSlots = { slot };
    return operation;
}

videocontrol::Operation sinkOp(int nodeId, int slot)
{
    videocontrol::Operation operation;
    operation.nodeId = nodeId;
    operation.kind = "sink";
    operation.inputs = { { slot } };
    operation.destination = "clip1/gen/level";
    return operation;
}
} // namespace

int main()
{
    {
        nlohmann::json wire = {
            { "version", 2 }, { "numSlots", 1 },
            { "signalWindow", { { "policy", "floor-frame-v1" },
                { "sampleRate", 48000 }, { "fps", 29.97 }, { "frameCount", 3 } } },
            { "operations", nlohmann::json::array({
                { { "kind", "signal.window" }, { "outputSlots", { 0 } },
                  { "frameValues", { 0.1, 0.3, 0.2 } } },
                { { "kind", "sink" }, { "inputs", { { 0 } } },
                  { "destination", "clip99/source/opacity" }, { "depth", 0.5 } }
            }) }
        };
        videocontrol::Plan plan;
        std::string error;
        check(videocontrol::parsePlanJson(wire, plan, error), "Signal window JSON admits");
        check(plan.signalWindow.sampleStart(1) == 1601
              && plan.signalWindow.sampleStart(2) == 3203, "fractional FPS floor windows");
        videocontrol::Executor executor;
        check(executor.bind(plan, error), "Signal window plan binds");
        const float expected[] { 0.05f, 0.15f, 0.1f };
        for (const int frame : { 2, 0, 1, 1, 0 })
        {
            executor.reset();
            const auto& values = executor.evaluate({}, {}, {}, 0, 0, frame, 29.97);
            check(values.size() == 1 && std::abs(values[0].value - expected[frame]) < 1.0e-7f,
                  "Signal window lookup survives repeated frame, seek and reset");
        }
        check(executor.evaluate({}, {}, {}, 0, 0, 3, 29.97).empty(), "out-of-range window fails closed");
        check(executor.evaluate({}, {}, {}, 0, 0, 0, 30.0).empty(), "mismatched FPS fails closed");
        auto invalid = wire;
        invalid["operations"][0]["frameValues"] = { 0.1, 0.2 };
        check(!videocontrol::parsePlanJson(invalid, plan, error), "partial columns rejected");
        invalid = wire;
        invalid["operations"][0]["frameValues"][1] = nullptr;
        check(!videocontrol::parsePlanJson(invalid, plan, error), "nonnumeric values rejected");
        invalid = wire;
        invalid["signalWindow"]["sampleRate"] = 44100;
        check(!videocontrol::parsePlanJson(invalid, plan, error), "wrong sample rate rejected");
        invalid = wire;
        invalid["signalWindow"]["policy"] = "first-sample-only";
        check(!videocontrol::parsePlanJson(invalid, plan, error), "unknown window policy rejected");
        invalid = wire;
        invalid["signalWindow"]["frameCount"] = "three";
        check(!videocontrol::parsePlanJson(invalid, plan, error), "invalid field type fails without throwing");
        invalid = wire;
        invalid["operations"][0]["frameValues"] = std::vector<float>(videocontrol::kMaxSignalFrameValues + 1);
        check(!videocontrol::parsePlanJson(invalid, plan, error), "frame value capacity rejected");
        invalid = wire;
        invalid["version"] = 1;
        check(!videocontrol::parsePlanJson(invalid, plan, error), "legacy version cannot hide window data");
        check(videocontrol::parsePlanJson(wire, plan, error), "valid plan still parses after refusals");
        plan.numSlots = 2;
        plan.operations.push_back(valueOp(77, 0.625f, 1));
        plan.operations.push_back(sinkOp(78, 1));
        check(executor.bind(plan, error), "mixed Signal and Control plan binds");
        for (const auto& request : { std::pair<int, double>{ 0, 30.0 }, { 3, 29.97 }, { -1, 29.97 } })
        {
            const auto& remaining = executor.evaluate({}, {}, {}, 0, 0, request.first, request.second);
            check(remaining.size() == 1 && remaining[0].destination == "clip1/gen/level"
                  && remaining[0].value == 0.625f, "missing Signal window preserves Control-only sink");
        }
        plan.operations.front().frameValues[0] = std::numeric_limits<float>::infinity();
        check(!videocontrol::validatePlan(plan, error), "nonfinite in-memory frame values rejected");
    }

    arbitmod::Score score;
    arbitmod::Clock clock;
    arbitmod::Audio audio;

    {
        videocontrol::Plan plan;
        plan.numSlots = 3;
        plan.operations.push_back(valueOp(1, 0.25f, 0));
        plan.operations.push_back(valueOp(2, 0.5f, 1));
        videocontrol::Operation math;
        math.nodeId = 3;
        math.kind = "control.math";
        math.inputs = { { 0, 1 }, {} };
        math.params = { 0.0f };
        math.outputSlots = { 2 };
        plan.operations.push_back(math);
        auto sink = sinkOp(4, 2);
        sink.depth = 0.5f;
        plan.operations.push_back(sink);

        videocontrol::Executor executor;
        std::string error;
        check(executor.bind(plan, error), "fan-in plan admits");
        const auto& values = executor.evaluate(score, clock, audio, 1.0f / 60.0f,
                                                1.0f / 60.0f);
        check(values.size() == 1, "fan-in plan emits one sink");
        check(values.size() == 1 && std::abs(values[0].value - 0.375f) < 1.0e-6f,
              "fan-in Math result reaches sink with depth");
    }

    {
        videocontrol::Plan plan;
        plan.numSlots = 2;
        plan.operations.push_back(valueOp(1, 0.8f, 0));
        videocontrol::Operation history;
        history.nodeId = 2;
        history.kind = "control.history";
        history.inputs = { { 0 } };
        history.params = { 0.2f };
        history.outputSlots = { 1 };
        plan.operations.push_back(history);
        plan.operations.push_back(sinkOp(3, 1));

        videocontrol::Executor executor;
        std::string error;
        check(executor.bind(plan, error), "History plan admits");
        const auto first = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(first.size() == 1 && std::abs(first[0].value - 0.2f) < 1.0e-6f,
              "History emits initial value first");
        const auto second = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(second.size() == 1 && std::abs(second[0].value - 0.8f) < 1.0e-6f,
              "History emits previous input next");
        executor.reset();
        const auto reset = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(reset.size() == 1 && std::abs(reset[0].value - 0.2f) < 1.0e-6f,
              "History reset is deterministic");
    }

    {
        videocontrol::Plan invalid;
        invalid.numSlots = static_cast<int>(videocontrol::kMaxSlots + 1);
        videocontrol::Executor executor;
        std::string error;
        check(!executor.bind(invalid, error), "over-cap plan rejects atomically");
        check(!error.empty(), "over-cap rejection carries a diagnostic");
    }

    {
        videocontrol::Plan invalid;
        invalid.numSlots = 1;
        invalid.operations.push_back(valueOp(1, 0.5f, 0));
        auto sink = sinkOp(2, 0);
        sink.targetClipId = 9;
        invalid.operations.push_back(sink);
        videocontrol::Executor executor;
        std::string error;
        check(!executor.bind(invalid, error), "partial structured target rejects atomically");
    }

    {
        audio.rms = 0.9f;
        audio.sourceCount = 1;
        audio.sources[0] = { 42, 1, 0.35f, 0.6f, 1.0f };
        videocontrol::Plan plan;
        plan.numSlots = 1;
        videocontrol::Operation source;
        source.nodeId = 1;
        source.kind = "source";
        source.source.type = arbitmod::SourceType::AudioRms;
        source.source.trackId = 42;
        source.outputSlots = { 0 };
        plan.operations.push_back(source);
        plan.operations.push_back(sinkOp(2, 0));
        videocontrol::Executor executor;
        std::string error;
        check(executor.bind(plan, error), "typed track-audio plan admits");
        const auto values = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(values.size() == 1 && std::abs(values[0].value - 0.35f) < 1.0e-6f,
              "track audio source selects stable track id instead of master");
        source.source.trackId = 99;
        plan.operations[0] = source;
        check(executor.bind(plan, error), "missing track-audio plan remains executable");
        const auto missing = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(missing.size() == 1 && missing[0].value == 0.0f,
              "missing track audio source fails closed to zero");
    }

    {
        // LFONode quarter-cycle fixtures after rate, depth, and phase fan-in.
        const float waves[4][4] = {
            { 1.0f, 0.0f, -1.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f, -1.0f },
            { -0.5f, 0.0f, 0.5f, -1.0f },
            { 1.0f, -1.0f, -1.0f, 1.0f }
        };
        for (int shape = 0; shape < 4; ++shape)
        {
            videocontrol::Plan plan;
            plan.numSlots = 4;
            plan.operations = { valueOp(1, 0.25f, 0), valueOp(2, 0.2f, 1),
                                valueOp(3, 0.125f, 2) };
            videocontrol::Operation lfo;
            lfo.nodeId = 4;
            lfo.kind = "source";
            lfo.source.type = arbitmod::SourceType::Lfo;
            lfo.inputs = { { 0 }, { 1 }, { 2 } };
            lfo.params.resize(26, 0.0f);
            lfo.params[21] = 1.0f;
            lfo.params[22] = 0.5f;
            lfo.params[23] = static_cast<float>(shape);
            lfo.params[24] = 0.4f;
            lfo.params[25] = 0.125f;
            lfo.outputSlots = { 3 };
            plan.operations.push_back(lfo);
            plan.operations.push_back(sinkOp(5, 3));
            videocontrol::Executor executor;
            std::string error;
            check(executor.bind(plan, error), "unipolar LFO plan admits");
            for (const float bpm : { 60.0f, 137.0f, 240.0f })
                for (int frame = 0; frame < 8; ++frame)
                {
                    const int quarter = frame < 4 ? frame : 7 - frame;
                    clock = { static_cast<float>(quarter) * 0.25f, bpm, 4.0f };
                    const auto values = executor.evaluate(score, clock, audio, 0.0f, 0.0f);
                    check(values.size() == 1 && std::abs(values[0].value
                        - (0.5f + 0.3f * waves[shape][quarter])) < 1.0e-6f,
                        "legacy waveform follows beat position and named modulation through BPM changes and seeks");
                }
            executor.reset();
            clock.beat = -0.5f;
            const auto negative = executor.evaluate(score, clock, audio, 1.0f, 1.0f);
            check(negative.size() == 1 && std::abs(negative[0].value
                - (0.5f + 0.3f * waves[shape][2])) < 1.0e-6f,
                "legacy LFO resets and wraps negative beat phase deterministically");

            // Mode zero continues to use the canonical bipolar descriptor.
            plan.operations[3].params[21] = 0.0f;
            plan.operations[3].source.lfo.periodBeats = 3.0f;
            plan.operations[3].source.lfo.hz = true;
            plan.operations[3].source.lfo.rateHz = 2.0f;
            check(executor.bind(plan, error), "canonical LFO plan still admits");
            for (const float bpm : { 60.0f, 137.0f, 240.0f })
            {
                clock = { 0.375f, bpm, 4.0f };
                const auto values = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
                check(values.size() == 1 && std::abs(values[0].value
                    - arbitmod::evaluateSource(plan.operations[3].source, score, clock, audio)) < 1.0e-6f,
                    "canonical mode preserves descriptor evaluation regardless of legacy parameters and inputs");
            }
        }
    }

    {
        videocontrol::Plan plan;
        plan.numSlots = 3;
        videocontrol::Operation gate;
        gate.nodeId = 1;
        gate.kind = "source";
        gate.source.type = arbitmod::SourceType::AudioRms;
        gate.outputSlots = { 0 };
        plan.operations.push_back(gate);
        videocontrol::Operation env;
        env.nodeId = 2;
        env.kind = "source";
        env.source.type = arbitmod::SourceType::Env;
        env.inputs = { { 0 } };
        env.params.resize(25, 0.0f);
        env.params[21] = 1.0f;
        env.params[22] = 1000.0f;
        env.params[23] = 250.0f;
        env.params[24] = 1000.0f;
        env.outputSlots = { 1 };
        plan.operations.push_back(env);
        env.nodeId = 3;
        env.params[22] = 2000.0f;
        env.outputSlots = { 2 };
        plan.operations.push_back(env);
        plan.operations.push_back(sinkOp(4, 1));
        plan.operations.push_back(sinkOp(5, 2));
        videocontrol::Executor executor, isolated;
        std::string error;
        check(executor.bind(plan, error), "gate A/H/R plan admits");
        check(isolated.bind(plan, error), "independent envelope executor admits");
        const auto frame = [&] (float beat, float bpm, float gateValue, float first, float second)
        {
            clock = { beat, bpm, 4.0f };
            audio.rms = gateValue;
            // Deliberately unrelated caller deltas: EnvNode owns its playhead delta.
            const auto values = executor.evaluate(score, clock, audio, 9.0f, 9.0f);
            check(values.size() == 2 && std::abs(values[0].value - first) < 1.0e-6f
                && std::abs(values[1].value - second) < 1.0e-6f,
                "gate envelope matches attack hold release and per-node timing state");
        };
        frame(0.0f, 120.0f, 1.0f, 0.0f, 0.0f);
        frame(0.5f, 120.0f, 1.0f, 0.25f, 0.125f);
        frame(1.0f, 60.0f, 1.0f, 0.75f, 0.375f);
        frame(1.25f, 120.0f, 0.00005f, 0.75f, 0.375f);
        frame(1.5f, 120.0f, 0.0f, 0.75f, 0.375f);
        frame(1.75f, 120.0f, 0.0f, 0.625f, 0.25f);
        frame(1.75f, 120.0f, 0.0f, 0.625f, 0.25f);
        frame(1.0f, 120.0f, 0.0f, 0.625f, 0.25f);
        frame(1.25f, 120.0f, 0.0f, 0.5f, 0.125f);
        audio.rms = 1.0f;
        const auto independent = isolated.evaluate(score, clock, audio, 0.0f, 0.0f);
        check(independent.size() == 2 && independent[0].value == 0.0f
            && independent[1].value == 0.0f, "executor state is independent");
        executor.reset();
        frame(1.25f, 120.0f, 1.0f, 0.0f, 0.0f);
        frame(20.0f, 120.0f, 1.0f, 1.0f, 0.5f);
        frame(21.0f, 120.0f, 0.0f, 1.0f, 0.5f);
        frame(21.5f, 120.0f, 0.0f, 0.75f, 0.25f);
        frame(22.0f, 0.0f, 0.0f, 0.5f, 0.0f);
        check(executor.bind(plan, error), "rebinding envelope plan resets state");
        frame(30.0f, 120.0f, 1.0f, 0.0f, 0.0f);

        plan.operations[1].params[22] = 0.0f;
        plan.operations[1].params[23] = 0.0f;
        plan.operations[1].params[24] = 0.0f;
        check(executor.bind(plan, error), "zero-time envelope plan admits");
        frame(0.0f, 120.0f, 1.0f, 1.0f, 0.0f);
        frame(0.0f, 120.0f, 0.0f, 0.0f, 0.0f);

        plan.operations[1].params[21] = 0.0f;
        check(executor.bind(plan, error), "canonical envelope plan still admits");
        score.notes.push_back(arbitmod::Note {});
        for (const float beat : { 0.025f, 0.5f, 1.05f, 2.0f })
        {
            clock.beat = beat;
            const auto canonical = executor.evaluate(score, clock, audio, 0.0f, 0.0f);
            check(canonical.size() == 2 && canonical[0].value
                == arbitmod::evaluateSource(plan.operations[1].source, score, clock, audio),
                "canonical envelope keeps score ADSR attack sustain release and silence semantics");
        }
    }

    if (failures != 0) return 1;
    std::cout << "video control plan: all checks passed\n";
    return 0;
}
