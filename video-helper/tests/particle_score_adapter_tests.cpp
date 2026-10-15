// Historical score adapters for particle bodies. A note's emission credit births
// bodies that a nonnegative Emission rate would hold back, and a sounding note
// attracts every live body toward its pitch point at mid-height. Every check
// replays the same notes, so a seek must reproduce the same bodies exactly.
#include "../src/particle_body_replay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace videorender;

namespace
{
int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

// A note at 120 BPM on the spawn track. Beat 2 is 1.0 s, and the note sounds for
// lengthBeats beats after that.
arbitmod::Note makeNote(int id, float midi, float startBeat, float lengthBeats, float velocity = 127.0f)
{
    arbitmod::Note note;
    note.id = id;
    note.midiNote = midi;
    note.velocity = velocity;
    note.startBeat = startBeat;
    note.lengthBeats = lengthBeats;
    note.freqHz = 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f);
    return note;
}

std::shared_ptr<ParticleHistorySource> makeHistory(std::vector<arbitmod::Note> notes)
{
    auto score = std::make_shared<arbitmod::Score>();
    score->scoreRevision = 1;
    score->notes = std::move(notes);
    auto history = std::make_shared<ParticleHistorySource>();
    history->score = score;
    history->timeline.reset(120, 4);
    auto& key = history->key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = key.backendGeneration = key.deviceGeneration = 1;
    key.scoreGeneration = key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
    return history;
}

// One body, no initial velocity, no links, no collisions, and no forces except
// the score adapters under test.
ParticleParams baseParams(const std::shared_ptr<ParticleHistorySource>& history)
{
    ParticleParams p;
    p.motionMode = 2;
    p.simulationSpace = 1;
    p.historicalReplay = true;
    p.count = 1;
    p.geometryCount = 0;
    p.lifetime = 2;
    p.force = 0;
    p.bodyRadius = 0.02f;
    p.history = history;
    return p;
}

ParticleSolidState sample(ParticleParams p, double seconds)
{
    std::string error;
    ParticleSolidState out;
    p.historyProjectSeconds = seconds;
    replayParticleBodies(p, seconds, 1, nullptr, &error, &out);
    if (!error.empty())
    {
        std::printf("replay error: %s\n", error.c_str());
        ++failures;
    }
    return out;
}

// Replays must match bit for bit, so compare the stored floats exactly.
bool sameBodies(const ParticleSolidState& a, const ParticleSolidState& b)
{
    if (a.count != b.count) return false;
    for (int i = 0; i < a.count; ++i)
        if (a.bodies[i].x != b.bodies[i].x || a.bodies[i].y != b.bodies[i].y
            || a.bodies[i].z != b.bodies[i].z || a.bodies[i].vx != b.bodies[i].vx
            || a.bodies[i].vy != b.bodies[i].vy || a.bodies[i].vz != b.bodies[i].vz)
            return false;
    return true;
}

// Pitch 60 maps to x 0.42 on the birth span. Pitch 60 births at that x too, so
// an attractor on the same note pulls only in Y.
void checkAttractorPullsUpward()
{
    const auto history = makeHistory({ makeNote(7001, 60.0f, 2.0f, 4.0f) });
    const auto plain = sample(baseParams(history), 1.5);
    auto attracting = baseParams(history);
    attracting.scoreAttractGain = 2.0f;
    const auto pulled = sample(attracting, 1.5);
    check(plain.count == 1 && pulled.count == 1 && plain.bodies[0].y == 0.12f,
          "a note-born body starts at the birth height with no attractor");
    check(pulled.count == 1 && pulled.bodies[0].y > plain.bodies[0].y
          && std::abs(pulled.bodies[0].x - plain.bodies[0].x) < 1e-5,
          "a sounding note pulls its body upward toward mid-height and keeps its pitch x");
}

// Attractor strength is gain times normalized velocity, so a softer note pulls less.
void checkAttractorFollowsVelocity()
{
    const auto loud = makeHistory({ makeNote(7001, 60.0f, 2.0f, 4.0f, 127.0f) });
    const auto soft = makeHistory({ makeNote(7001, 60.0f, 2.0f, 4.0f, 64.0f) });
    auto loudParams = baseParams(loud), softParams = baseParams(soft);
    loudParams.scoreAttractGain = softParams.scoreAttractGain = 2.0f;
    const auto loudBody = sample(loudParams, 1.5), softBody = sample(softParams, 1.5);
    check(loudBody.count == 1 && softBody.count == 1
          && loudBody.bodies[0].y - 0.12f > softBody.bodies[0].y - 0.12f
          && softBody.bodies[0].y > 0.12f,
          "a louder sounding note attracts its body harder than a softer one");
}

// Seeks to other times and back reproduce the attracted bodies exactly.
void checkAttractorSeekReproduces()
{
    const auto history = makeHistory({ makeNote(7001, 60.0f, 2.0f, 4.0f) });
    auto attracting = baseParams(history);
    attracting.scoreAttractGain = 2.0f;
    const auto direct = sample(attracting, 1.5);
    sample(attracting, 0.9);
    sample(attracting, 2.2);
    sample(attracting, 1.2);
    check(sameBodies(direct, sample(attracting, 1.5)),
          "a seek and replay of a score attractor reproduces identical bodies");
}

// With Emission rate zero, a body is held back until budget accrues. An onset's
// credit releases it at once, and a credit below one body does not.
void checkEmissionCreditBirthsAtOnset()
{
    const auto history = makeHistory({ makeNote(7001, 60.0f, 2.0f, 4.0f) });
    auto gated = baseParams(history);
    gated.emissionRate = 0.0f;
    auto credited = gated;
    credited.scoreEmitGain = 1.5f;
    auto partial = gated;
    partial.scoreEmitGain = 0.5f;
    const auto held = sample(gated, 1.5), burst = sample(credited, 1.5), fractional = sample(partial, 1.5);
    check(held.count == 0, "a zero emission rate holds back a note-born body with no credit");
    check(burst.count == 1 && fractional.count == 0,
          "an onset credit of one body releases the body; a fractional credit does not");
    sample(credited, 0.9);
    sample(credited, 2.2);
    check(sameBodies(burst, sample(credited, 1.5)),
          "emission credit replays the same bodies after seeks away from the onset");
}

// Immediate emission ignores the budget, so an emission credit changes nothing.
void checkImmediateEmissionIgnoresCredit()
{
    const auto history = makeHistory({ makeNote(7001, 60.0f, 2.0f, 4.0f) });
    auto plain = baseParams(history);
    auto credited = plain;
    credited.scoreEmitGain = 1.5f;
    check(sameBodies(sample(plain, 1.5), sample(credited, 1.5)),
          "an immediate emission rate is unchanged by score emission credit");
}

// Adapter events share the replay's 512-event cap. A window of 520 notes is
// refused only when the emission adapter is on, and the refusal names it.
void checkAdapterEventsShareCap()
{
    std::vector<arbitmod::Note> crowd;
    for (int i = 0; i < 520; ++i)
        crowd.push_back(makeNote(9000 + i, 60.0f, 0.01f + 0.03f * float(i), 0.01f));
    const auto history = makeHistory(crowd);
    auto packed = baseParams(history);
    packed.lifetime = 10;
    packed.geometryCount = 1;
    packed.historyProjectSeconds = 9.0;
    auto packedEmit = packed;
    packedEmit.scoreEmitGain = 1.0f;
    const auto window = particleReplayWindow(packedEmit, 9.0);
    const auto noop = [](std::int64_t, const ParticleScoreEvent&) {};
    check(visitParticleScoreEvents(packed, window, noop, 0.0, 0.0, 0.0)
          && !visitParticleScoreEvents(packedEmit, window, noop, 0.0, 1.0, 0.0),
          "520 weighted onsets exceed the shared 512-event cap only with a positive emission gain");
    check(particleEventCapDiagnostic(packedEmit, window, "generic cap").find("Score note impulse") != std::string::npos,
          "the cap refusal names score note adapters when one is active");
    check(particleEventCapDiagnostic(packed, window, "generic cap") == "generic cap",
          "the cap refusal without adapters keeps its generic diagnostic");
}
}

int main()
{
    checkAttractorPullsUpward();
    checkAttractorFollowsVelocity();
    checkAttractorSeekReproduces();
    checkEmissionCreditBirthsAtOnset();
    checkImmediateEmissionIgnoresCredit();
    checkAdapterEventsShareCap();
    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
