// Harmony feed: score.harmony parsing, per-beat uniform values, the
// score.harmony.signal control node and the prelude declarations.

#include "harmony_signals.h"
#include "harmony_signals_json.h"
#include "score_json_parser.h"
#include "shader_dialect.h"
#include "shader_generator.h"
#include "video_control_plan.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{
int failures = 0;

void check (bool condition, const std::string& message)
{
    if (! condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool near (double a, double b, double epsilon = 1.0e-5)
{
    return std::abs (a - b) <= epsilon;
}

nlohmann::json note (int id, double midi, double start, double length)
{
    return { { "id", id }, { "trackId", 0 }, { "startBeat", start }, { "lengthBeats", length },
             { "midiNote", midi }, { "velocity", 100.0 },
             { "freqHz", 440.0 * std::exp2 ((midi - 69.0) / 12.0) } };
}

nlohmann::json chordSegment (double start, double end, int pc, int quality, double tension)
{
    return { { "startBeat", start }, { "endBeat", end }, { "rootPc", pc },
             { "rootPos", pc / 12.0 }, { "rootCents", pc * 100.0 },
             { "qualityClass", quality }, { "tension", tension } };
}

// Song A: C, F, G, A minor, one chord per 4/4 bar, quarter-note rhythm.
// Song B: the same rhythm, but a static cluster [84,85,86] over root 48.
nlohmann::json songScore (bool withHarmony, bool songA)
{
    nlohmann::json score = { { "schemaVersion", 2 }, { "rootFreq", 261.625565 } };
    nlohmann::json notes = nlohmann::json::array();
    const int chords[4][3] = { { 60, 64, 67 }, { 65, 69, 72 }, { 67, 71, 74 }, { 69, 72, 76 } };
    int id = 1;
    for (int bar = 0; bar < 4; ++bar)
        for (int beat = 0; beat < 4; ++beat)
        {
            const double start = bar * 4.0 + beat;
            if (songA)
                for (int voice = 0; voice < 3; ++voice)
                    notes.push_back (note (id++, chords[bar][voice], start, 1.0));
            else
                for (const int midi : { 48, 84, 85, 86 })
                    notes.push_back (note (id++, midi, start, 1.0));
        }
    score["notes"] = notes;
    if (withHarmony)
    {
        nlohmann::json harmony = { { "version", 1 }, { "phraseBars", 4 }, { "songEndBeat", 16.0 } };
        if (songA)
            harmony["chords"] = { chordSegment (0, 4, 0, arbitharmony::QualityMajor, 0.33),
                                  chordSegment (4, 8, 5, arbitharmony::QualityMajor, 0.33),
                                  chordSegment (8, 12, 7, arbitharmony::QualityMajor, 0.33),
                                  chordSegment (12, 16, 9, arbitharmony::QualityMinor, 0.35) };
        else
            harmony["chords"] = { chordSegment (0, 16, 0, arbitharmony::QualityOther, 0.66) };
        harmony["keys"] = { { { "startBeat", 0.0 }, { "endBeat", 16.0 }, { "tonicPc", 0 }, { "mode", 0 } } };
        score["harmony"] = harmony;
    }
    return score;
}

arbitharmony::HarmonyTimeline parse (const nlohmann::json& scoreObject, arbitmod::Score& score)
{
    std::string error;
    check (videohelper::scorejson::parseScoreJson (scoreObject, score, error),
           "score payload parses: " + error);
    return arbitharmony::parseHarmonyJson (scoreObject, score);
}

videotime::BeatTimeline fourFour()
{
    videotime::BeatTimeline timeline;
    timeline.reset (120.0, 4.0);
    return timeline;
}

videocontrol::Plan signalPlan (float selector)
{
    videocontrol::Plan plan;
    plan.numSlots = 1;
    videocontrol::Operation signal;
    signal.nodeId = 1;
    signal.kind = "score.harmony.signal";
    signal.params = { selector };
    signal.outputSlots = { 0 };
    videocontrol::Operation sink;
    sink.nodeId = 2;
    sink.kind = "sink";
    sink.inputs = { { 0 } };
    sink.destination = "clip1/gen/level";
    plan.operations = { signal, sink };
    return plan;
}

// One frame through the executor, with the clock the export and viewport
// loops build: the frame's beat from the timeline, rounded to float.
float triggerAt (videocontrol::Executor& executor, const arbitmod::Score& score,
                 const arbitharmony::HarmonyTimeline& harmony, const videotime::BeatTimeline& timeline,
                 std::int64_t frame, double fps)
{
    const double seconds = static_cast<double> (frame) / fps;
    arbitmod::Clock clock;
    clock.beat = static_cast<float> (timeline.secondsToBeat (seconds));
    const double dtBeats = timeline.secondsToBeat (seconds + 1.0 / fps) - timeline.secondsToBeat (seconds);
    executor.setHarmonyContext (&harmony, &timeline);
    const auto& sinks = executor.evaluate (score, clock, {}, static_cast<float> (dtBeats),
                                           static_cast<float> (1.0 / fps), frame, fps);
    return sinks.size() == 1 ? sinks[0].value : -1.0f;
}

void checkPayloadValues()
{
    arbitmod::Score score;
    const auto harmony = parse (songScore (true, true), score);
    const auto timeline = fourFour();
    check (harmony.present, "harmony object is present");
    check (harmony.chords.size() == 4 && harmony.keys.size() == 1, "segments parse");

    struct Expected { double beat; float root; float quality; float tension; float age; };
    const Expected expected[] = {
        { 0.0, 0.0f, 1.0f, 0.33f, 0.0f }, { 2.5, 0.0f, 1.0f, 0.33f, 2.5f },
        { 4.0, 5.0f / 12.0f, 1.0f, 0.33f, 0.0f }, { 9.0, 7.0f / 12.0f, 1.0f, 0.33f, 1.0f },
        { 15.5, 9.0f / 12.0f, 2.0f, 0.35f, 3.5f } };
    for (const auto& item : expected)
    {
        const auto signals = arbitharmony::evaluateHarmony (harmony, timeline, item.beat);
        const std::string at = " at beat " + std::to_string (item.beat);
        check (near (signals.chord[0], item.root), "uChord.x is the payload root" + at);
        check (near (signals.chord[1], item.quality), "uChord.y is the payload quality" + at);
        check (near (signals.chord[2], item.tension), "uChord.z is the payload tension" + at);
        check (near (signals.chord[3], item.age), "uChord.w counts beats since the change" + at);
        check (near (signals.key[0], 0.0) && near (signals.key[1], 1.0), "uKey is C Ionian" + at);
        check (near (signals.key[3], 5.0 / 6.0), "Ionian brightness" + at);
    }

    const auto mid = arbitharmony::evaluateHarmony (harmony, timeline, 6.0);
    check (near (mid.phrase[0], 1.0) && near (mid.phrase[1], 0.0), "bar 1 sits in phrase 0");
    check (near (mid.phrase[2], 1.5 / 4.0), "phrase phase counts bars and the bar phase");
    check (near (mid.phrase[3], 6.0 / 16.0), "song progress follows songEndBeat");
    check (near (mid.lastOnsetBeat, 6.0), "uLastOnsetBeat is the latest onset");
    const auto late = arbitharmony::evaluateHarmony (harmony, timeline, 18.0);
    check (near (late.phrase[0], 4.0) && near (late.phrase[1], 1.0), "bar 4 opens phrase 1");
    check (near (late.phrase[2], 0.5 / 4.0), "phrase phase restarts in phrase 1");
    check (near (late.phrase[3], 1.0), "song progress clamps at the end");
    check (near (late.chord[3], 6.0), "the last chord holds after the song ends");
}

void checkAbControl()
{
    arbitmod::Score scoreA, scoreB;
    const auto a = parse (songScore (true, true), scoreA);
    const auto b = parse (songScore (true, false), scoreB);
    const auto timeline = fourFour();
    for (int bar = 0; bar < 4; ++bar)
    {
        const std::string at = " in bar " + std::to_string (bar);
        for (const double offset : { 0.0, 1.0, 3.5 })
        {
            const double beat = bar * 4.0 + offset;
            const auto sa = arbitharmony::evaluateHarmony (a, timeline, beat);
            const auto sb = arbitharmony::evaluateHarmony (b, timeline, beat);
            bool differs = false;
            for (int i = 0; i < 3; ++i)
                differs = differs || ! near (sa.chord[i], sb.chord[i]);
            check (differs, "uChord differs between song A and song B" + at);
            check (near (sa.chord[3], offset), "song A's change counter resets each bar" + at);
            check (near (sb.chord[3], beat), "song B's change counter never resets" + at);
        }
    }

    // The chord-change trigger over both songs, frame by frame like export:
    // song A pulses on each bar line, song B only on the beat-0 prime.
    const double fps = 30.0;
    std::string error;
    videocontrol::Executor triggerA, triggerB;
    check (triggerA.bind (signalPlan (arbitharmony::SignalChordChange), error), "song A trigger binds: " + error);
    check (triggerB.bind (signalPlan (arbitharmony::SignalChordChange), error), "song B trigger binds: " + error);
    std::vector<std::int64_t> pulsesA, pulsesB;
    for (std::int64_t frame = 0; frame < 240; ++frame)
    {
        if (triggerAt (triggerA, scoreA, a, timeline, frame, fps) > 0.5f) pulsesA.push_back (frame);
        if (triggerAt (triggerB, scoreB, b, timeline, frame, fps) > 0.5f) pulsesB.push_back (frame);
    }
    check (pulsesA == std::vector<std::int64_t> { 0, 60, 120, 180 }, "song A's trigger fires on every bar line");
    check (pulsesB == std::vector<std::int64_t> { 0 }, "song B's trigger fires only on the beat-0 prime");
}

void checkOldPayload()
{
    arbitmod::Score score;
    const auto harmony = parse (songScore (false, true), score);
    const auto timeline = fourFour();
    check (! harmony.present, "a payload without harmony parses as absent");
    check (score.notes.size() == 48, "the old payload's notes still load");
    for (const double beat : { 0.0, 4.0, 7.5, 15.0 })
    {
        const auto signals = arbitharmony::evaluateHarmony (harmony, timeline, beat);
        bool zero = true;
        for (int i = 0; i < 4; ++i)
            zero = zero && signals.chord[i] == 0.0f && signals.key[i] == 0.0f && signals.phrase[i] == 0.0f;
        check (zero, "an old payload keeps uChord, uKey and uPhrase at zero");
        check (near (signals.lastOnsetBeat, std::floor (beat)), "onsets still come from the notes");
    }

    // Malformed entries are dropped one by one; the score never fails.
    auto broken = songScore (true, true);
    broken["harmony"]["chords"].push_back ({ { "startBeat", "x" } });
    broken["harmony"]["chords"].push_back ({ { "startBeat", 5.0 }, { "endBeat", 1.0 } });
    broken["harmony"]["keys"] = "not an array";
    arbitmod::Score brokenScore;
    const auto tolerant = parse (broken, brokenScore);
    check (tolerant.present && tolerant.chords.size() == 4 && tolerant.keys.empty(),
           "malformed harmony entries are skipped");
    arbitmod::Score other;
    nlohmann::json notObject = songScore (false, true);
    notObject["harmony"] = 42;
    check (! parse (notObject, other).present, "a non-object harmony value reads as absent");
}

void checkGapsAndMeters()
{
    arbitharmony::HarmonyTimeline harmony;
    harmony.present = true;
    harmony.chords = { { 0.0, 3.0, 0, 0.0, 0.0, arbitharmony::QualityMajor, 0.3 },
                       { 4.0, 8.0, 2, 2.0 / 12.0, 200.0, arbitharmony::QualityMinor, 0.4 } };
    const auto timeline = fourFour();
    const auto rest = arbitharmony::evaluateHarmony (harmony, timeline, 3.5);
    check (rest.chordIndex == 0 && near (rest.chord[3], 3.5), "a chord holds through the rest after it");
    const auto before = arbitharmony::evaluateHarmony (harmony, timeline, -1.0);
    check (before.chordIndex == -1 && before.chord[1] == 0.0f, "no chord before the first segment");

    videotime::BeatTimeline meters;
    meters.set ({ { 0.0, 120.0, true } }, { { 0.0, 4, 4 }, { 8.0, 3, 4 } });
    double phase = 0.0;
    check (arbitharmony::barIndexAtBeat (meters, 11.0, phase) == 3 && near (phase, 0.0),
           "bars count across a meter change");
    check (arbitharmony::barIndexAtBeat (meters, 12.5, phase) == 3 && near (phase, 1.5 / 3.0),
           "bar phase uses the 3/4 bar length");
}

void checkControlNode()
{
    arbitmod::Score score;
    const auto harmony = parse (songScore (true, true), score);
    const auto timeline = fourFour();

    const auto planFor = [] (float selector)
    {
        videocontrol::Plan plan;
        plan.numSlots = 1;
        videocontrol::Operation signal;
        signal.nodeId = 1;
        signal.kind = "score.harmony.signal";
        signal.params = { selector };
        signal.outputSlots = { 0 };
        videocontrol::Operation sink;
        sink.nodeId = 2;
        sink.kind = "sink";
        sink.inputs = { { 0 } };
        sink.destination = "clip1/gen/level";
        plan.operations = { signal, sink };
        return plan;
    };

    videocontrol::Executor trigger;
    std::string error;
    check (trigger.bind (planFor (arbitharmony::SignalChordChange), error), "harmony signal plan binds: " + error);
    const double fps = 30.0;
    const double dtBeats = 120.0 / 60.0 / fps;
    int pulses = 0;
    for (int frame = 0; frame < 240; ++frame)
    {
        arbitmod::Clock clock;
        clock.beat = static_cast<float> (frame * dtBeats);
        trigger.setHarmonyContext (&harmony, &timeline);
        const auto& sinks = trigger.evaluate (score, clock, {}, static_cast<float> (dtBeats),
                                              static_cast<float> (1.0 / fps), frame, fps);
        check (sinks.size() == 1, "one sink value per frame");
        if (! sinks.empty() && sinks[0].value > 0.5f)
        {
            ++pulses;
            check (frame % 60 == 0, "the chord-change trigger fires on a bar line");
        }
    }
    check (pulses == 4, "the chord-change trigger fires once per chord");

    videocontrol::Executor tension;
    check (tension.bind (planFor (arbitharmony::SignalChordTension), error), "tension plan binds");
    arbitmod::Clock clock;
    clock.beat = 13.0f;
    tension.setHarmonyContext (&harmony, &timeline);
    const auto tensionValue = tension.evaluate (score, clock, {}, 0.0f, 0.0f);
    check (tensionValue.size() == 1 && near (tensionValue[0].value, 0.35), "tension output matches the payload");

    videocontrol::Executor progress;
    check (progress.bind (planFor (arbitharmony::SignalSongProgress), error), "progress plan binds");
    clock.beat = 4.0f;
    progress.setHarmonyContext (&harmony, &timeline);
    const auto progressValue = progress.evaluate (score, clock, {}, 0.0f, 0.0f);
    check (progressValue.size() == 1 && near (progressValue[0].value, 0.25), "song progress output");

    videocontrol::Executor detached;
    check (detached.bind (planFor (arbitharmony::SignalChordRoot), error), "detached plan binds");
    clock.beat = 5.0f;
    const auto detachedValue = detached.evaluate (score, clock, {}, 0.0f, 0.0f);
    check (detachedValue.size() == 1 && detachedValue[0].value == 0.0f, "no harmony context reads 0");
}

void checkSeekTrigger()
{
    arbitmod::Score score;
    const auto harmony = parse (songScore (true, true), score);
    const auto timeline = fourFour();
    const double fps = 30.0;
    std::string error;

    // Export: every frame from 0.
    videocontrol::Executor exportRun;
    check (exportRun.bind (signalPlan (arbitharmony::SignalChordChange), error), "export trigger binds: " + error);
    std::vector<float> exported;
    for (std::int64_t frame = 0; frame < 240; ++frame)
        exported.push_back (triggerAt (exportRun, score, harmony, timeline, frame, fps));

    // Viewport: skips ahead, seeks back and loops. Every frame it renders
    // must read what export read at that frame.
    videocontrol::Executor viewport;
    check (viewport.bind (signalPlan (arbitharmony::SignalChordChange), error), "viewport trigger binds");
    const std::int64_t visits[] = { 0, 1, 2, 90, 91, 30, 31, 120, 121, 179, 180, 181, 61, 5, 239, 60, 0 };
    for (const auto frame : visits)
    {
        const float value = triggerAt (viewport, score, harmony, timeline, frame, fps);
        check (near (value, exported[static_cast<std::size_t> (frame)]),
               "the viewport trigger matches export at frame " + std::to_string (frame));
    }
    check (near (triggerAt (viewport, score, harmony, timeline, 90, fps), 0.0),
           "a jump into the middle of a chord does not fire");
    check (near (triggerAt (viewport, score, harmony, timeline, 120, fps), 1.0),
           "a jump onto a chord change fires, as export does");

    // Without frame numbers a backward or large jump re-primes: only a chord's
    // first beat fires.
    videocontrol::Executor unframed;
    check (unframed.bind (signalPlan (arbitharmony::SignalChordChange), error), "unframed trigger binds");
    const auto unframedAt = [&] (float beat)
    {
        arbitmod::Clock clock;
        clock.beat = beat;
        unframed.setHarmonyContext (&harmony, &timeline);
        const auto& sinks = unframed.evaluate (score, clock, {}, 1.0f / 15.0f, 1.0f / 30.0f);
        return sinks.size() == 1 ? sinks[0].value : -1.0f;
    };
    check (near (unframedAt (1.0f), 0.0), "unframed first evaluation mid-chord is quiet");
    check (near (unframedAt (6.0f), 0.0), "unframed forward jump into a chord is quiet");
    check (near (unframedAt (2.0f), 0.0), "unframed backward jump into a chord is quiet");
    check (near (unframedAt (4.0f), 1.0), "unframed jump onto a chord's first beat fires");
}

void checkShaderClockPacking()
{
    arbitmod::Score score;
    const auto harmony = parse (songScore (true, true), score);
    const auto timeline = fourFour();
    // 4.5 s at 120 BPM is beat 9: G major, one beat in, bar 2.
    const auto clock = videorender::makeShaderClock (4.5, 0.0, 10.0, timeline, 30.0, true, 135, &harmony);
    const auto signals = arbitharmony::evaluateHarmony (harmony, timeline, 9.0);
    check (near (clock.beat, 9.0), "the clock sits on beat 9");
    for (int i = 0; i < 4; ++i)
    {
        const std::string component = " component " + std::to_string (i);
        check (clock.chord[i] == signals.chord[i], "uChord packs the evaluated chord" + component);
        check (clock.key[i] == signals.key[i], "uKey packs the evaluated key" + component);
        check (clock.phrase[i] == signals.phrase[i], "uPhrase packs the evaluated phrase" + component);
    }
    check (near (clock.chord[0], 7.0 / 12.0) && near (clock.chord[1], 1.0)
           && near (clock.chord[2], 0.33) && near (clock.chord[3], 1.0), "uChord carries G major one beat in");
    check (near (clock.key[0], 0.0) && near (clock.key[1], 1.0) && near (clock.key[3], 5.0 / 6.0),
           "uKey carries C Ionian");
    check (near (clock.phrase[0], 2.0) && near (clock.phrase[3], 9.0 / 16.0), "uPhrase carries bar 2");
    check (near (clock.lastOnsetBeat, 9.0), "uLastOnsetBeat carries the latest onset");

    const auto bare = videorender::makeShaderClock (4.5, 0.0, 10.0, timeline, 30.0, true, 135);
    bool zero = bare.lastOnsetBeat == 0.0f;
    for (int i = 0; i < 4; ++i)
        zero = zero && bare.chord[i] == 0.0f && bare.key[i] == 0.0f && bare.phrase[i] == 0.0f;
    check (zero, "a clock without a harmony timeline keeps the harmony fields at zero");
}

std::size_t occurrences (const std::string& text, const std::string& needle)
{
    std::size_t count = 0;
    for (auto at = text.find (needle); at != std::string::npos; at = text.find (needle, at + 1)) ++count;
    return count;
}

bool listed (const std::vector<std::string>& names, const std::string& name)
{
    for (const auto& item : names) if (item == name) return true;
    return false;
}

void checkUserOwnedHarmonyNames()
{
    // A shader written for the media-machine effect prelude declares uChord
    // itself. It keeps its declaration, receives the values and keeps the
    // accessors, which are defined after its body.
    const auto own = arbitshader::wrapToContract (
        "uniform vec4 uChord;\n"
        "// uniform vec4 uKey; is only a comment\n"
        "float level(){ return uChord.z + arbitChordRoot(); }\n"
        "void main(){ fragColor = vec4(level(), uKey.x, 0.0, 1.0); }\n");
    check (own.ok && ! own.hasError(), "a shader that declares uChord itself still wraps");
    check (occurrences (own.glsl, "uChord;") == 1, "uChord is declared once, by the shader");
    check (listed (own.bodyHarmonyUniforms, "uChord") && own.foreignHarmonyUniforms.empty(),
           "a vec4 uChord declaration still receives the harmony values");
    check (occurrences (own.glsl, "uniform vec4  uKey;") == 1, "names the shader only mentions stay in the prelude");
    const auto prototype = own.glsl.find ("float arbitChordRoot();");
    const auto body = own.glsl.find ("float level()");
    const auto definition = own.glsl.find ("float arbitChordRoot(){ return uChord.x; }");
    check (prototype != std::string::npos && body != std::string::npos && definition != std::string::npos
           && prototype < body && body < definition,
           "chord accessors are prototyped in the prelude and defined after the shader");
    check (own.glsl.find ("float arbitKeyTonic(){ return uKey.x; }") < body,
           "accessors of prelude uniforms stay in the prelude");

    // The same uniform with another type is the shader's own: no harmony
    // values, no accessors that read it.
    const auto foreign = arbitshader::wrapToContract (
        "uniform float uKey;\nvoid main(){ fragColor = vec4(uKey); }\n");
    check (foreign.ok, "a shader that declares float uKey still wraps");
    check (listed (foreign.foreignHarmonyUniforms, "uKey"), "float uKey is foreign to the harmony feed");
    check (occurrences (foreign.glsl, "uKey;") == 1 && foreign.glsl.find ("arbitKeyTonic") == std::string::npos,
           "the prelude neither declares uKey nor reads it");

    // A parameter may use a harmony name, as before the harmony feed.
    const auto isf = arbitshader::wrapToContract (
        "/*{\"ISFVSN\":\"2\",\"INPUTS\":[{\"NAME\":\"uPhrase\",\"TYPE\":\"float\",\"DEFAULT\":0.5}]}*/\n"
        "void main(){ gl_FragColor = vec4(uPhrase); }\n");
    check (isf.ok && ! isf.hasError(), "an ISF INPUT named uPhrase is accepted");
    check (occurrences (isf.glsl, "uPhrase;") == 1 && isf.glsl.find ("arbitBarIndex") == std::string::npos,
           "the ISF input owns uPhrase");
    check (listed (isf.foreignHarmonyUniforms, "uPhrase"), "the renderer leaves the ISF uPhrase alone");

    const auto param = arbitshader::wrapToContract (
        "// ARBIT_PARAM {\"NAME\":\"uLastOnsetBeat\",\"TYPE\":\"float\",\"DEFAULT\":0.0}\n"
        "void main(){ fragColor = vec4(uLastOnsetBeat); }\n");
    check (param.ok && ! param.hasError(), "an ARBIT_PARAM named uLastOnsetBeat is accepted");
    check (occurrences (param.glsl, "uLastOnsetBeat;") == 1 && param.glsl.find ("arbitLastOnsetAge") == std::string::npos,
           "the parameter owns uLastOnsetBeat");

    // A shader that defines an accessor name keeps its own definition.
    const auto accessor = arbitshader::wrapToContract (
        "float arbitChordRoot(){ return 0.5; }\nvoid main(){ fragColor = vec4(arbitChordRoot()); }\n");
    check (accessor.ok, "a shader that defines arbitChordRoot still wraps");
    check (occurrences (accessor.glsl, "float arbitChordRoot(") == 1, "arbitChordRoot is defined once");
    check (occurrences (accessor.glsl, "uniform vec4  uChord;") == 1
           && accessor.glsl.find ("float arbitChordTension(){") != std::string::npos,
           "the other chord accessors stay");

    // A comma list declaration is recognised too.
    const auto list = arbitshader::wrapToContract (
        "uniform vec4 uTint, uChord;\nvoid main(){ fragColor = uChord * uTint; }\n");
    check (occurrences (list.glsl, "uChord;") == 1 && listed (list.bodyHarmonyUniforms, "uChord"),
           "uChord declared in a uniform list is the shader's");
}

void checkPrelude()
{
    const auto prelude = arbitshader::arbitPrelude (arbitshader::Dialect::BareGlsl, nullptr, true);
    for (const char* needle : { "uniform vec4  uChord;", "uniform float uLastOnsetBeat;",
                                "uniform vec4  uKey;", "uniform vec4  uPhrase;",
                                "float arbitChordRoot()", "float arbitChordTension()",
                                "float arbitChordPulse(float width)", "float arbitKeyBrightness()",
                                "float arbitPhrasePhase()", "float arbitSongProgress()",
                                "float arbitLastOnsetAge()" })
        check (prelude.find (needle) != std::string::npos, std::string ("prelude declares ") + needle);
    // Harmony names arrived after shaders that use them, so they are not
    // reserved; the prelude steps aside instead (checkUserOwnedHarmonyNames).
    for (const char* name : { "uChord", "uKey", "uPhrase", "uLastOnsetBeat", "arbitChordQuality", "arbitBarIndex" })
        check (! arbitshader::detail::isReservedName (name), std::string ("harmony name is not reserved: ") + name);
}
} // namespace

int main()
{
    checkPayloadValues();
    checkAbControl();
    checkOldPayload();
    checkGapsAndMeters();
    checkControlNode();
    checkSeekTrigger();
    checkShaderClockPacking();
    checkUserOwnedHarmonyNames();
    checkPrelude();
    if (failures != 0)
        return 1;
    std::cout << "harmony uniforms: all checks passed\n";
    return 0;
}
