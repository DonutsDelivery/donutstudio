#pragma once

#include "../src/water_surface_replay.h"
#include "../src/block_c_frame_owner.h"
#include "../../shared/GeometryTimelineSceneBinding.h"
#include "../../shared/SceneAovOperationContract.h"

#include <cstring>
#include <numeric>

namespace waterreplayfixture
{
inline videowire::geometry::GeometryData mesh(std::uint32_t columns, std::uint32_t rows)
{
    videowire::geometry::GeometryData result;
    for (std::uint32_t y = 0; y < rows; ++y)
        for (std::uint32_t x = 0; x < columns; ++x)
        {
            result.positions.push_back({static_cast<float>(x), static_cast<float>(y), 0});
            result.vertexIds.push_back(y * columns + x + 1);
        }
    for (std::uint32_t y = 0; y + 1 < rows; ++y)
        for (std::uint32_t x = 0; x + 1 < columns; ++x)
        {
            const auto a = y * columns + x, b = a + 1, c = a + columns, d = c + 1;
            for (const auto i : {a, b, c, b, d, c}) result.indices.push_back(i);
        }
    return result;
}

inline videowire::geometry::OperationRecord operation(double speed, double damping, double score, double audio)
{
    using namespace videowire::geometry;
    OperationRecord op;
    op.code = OperationCode::fieldWaterHeight;
    op.stableId = op.outputStableId = 10202; op.inputStableId = 10201;
    op.field.domain = Domain::vertex; op.field.valueType = ValueType::floatValue;
    op.field.values = {speed, damping, score, audio, 0, 0, 0, 0};
    return op;
}

inline videorender::ParticleHistorySource source()
{
    videorender::ParticleHistorySource result;
    auto score = std::make_shared<arbitmod::Score>(); score->scoreRevision = 1;
    result.score = score;
    auto& key = result.key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = key.backendGeneration = key.deviceGeneration = 1;
    key.scoreGeneration = key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
    key.fps = 30;
    result.timeline.reset(120, 4);
    return result;
}

inline std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frame(
    const videorender::ParticleHistorySource& source,double projectSeconds,
    double fps = 120, bool exporting = false)
{
    const auto& key = source.key;
    const canonicalblockc::OwnerIdentity identity {
        key.projectGeneration,key.sourceGeneration,key.helperGeneration,key.backendGeneration,
        key.deviceGeneration,key.scoreGeneration,key.beatMapGeneration,key.fpsGeneration,
        key.loopGeneration,key.seekGeneration
    };
    const auto target = std::llround(projectSeconds * fps);
    canonicalblockc::FrameOwner owner;
    const auto beatAt = [&](std::int64_t index) {
        return source.timeline.secondsToBeat(static_cast<double>(index) / fps);
    };
    return exporting
        ? owner.frameAt(canonicalblockc::ExportFrameRequest{identity,target,fps},source.score,beatAt)
        : owner.frameAt(canonicalblockc::PreviewFrameRequest{identity,target,fps},source.score,beatAt);
}

inline arbitmod::Note note(int id, float startBeat, float pitch = 0, float velocity = 127)
{
    arbitmod::Note result;
    result.id = id; result.startBeat = startBeat; result.midiNote = pitch; result.velocity = velocity;
    return result;
}

inline std::vector<arbitblockb::FeatureFrame> audio(std::size_t count)
{
    std::vector<arbitblockb::FeatureFrame> result(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        result[i].frameIndex = static_cast<std::int64_t>(i);
        result[i].sampleIndex = static_cast<std::int64_t>(i + 1) * arbitblockb::kHop;
    }
    return result;
}

inline bool near(double a, double b) { return std::abs(a - b) <= 1e-9 * std::max(1.0, std::abs(b)); }
inline bool zero(const std::vector<double>& v)
{
    return std::all_of(v.begin(), v.end(), [](double value) { return value == 0; });
}
inline double total(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0); }
}

// The prepared step API shares the ordinary replay authority. Independent
// stencil/momentum checks below keep equivalence from being the only oracle.
template <typename Check>
inline void waterSurfacePreparedSteps(Check check)
{
    using namespace videorender;
    using namespace videowire::geometry;
    namespace fixture = waterreplayfixture;
    constexpr double dt = 1.0 / 120;
    std::string error;
    auto history = fixture::source();
    auto frames = fixture::audio(1200);
    frames[0].onset = 1; frames[0].onsetAge = 0;
    history.audioFrames = &frames; history.audioSampleRate = 61440;
    const auto rest = fixture::mesh(3,3);
    WaterSurfaceGrid grid;
    check(validateWaterSurfaceGrid(rest,grid,error), "prepared water fixture admits exact rest grid");
    const auto op = fixture::operation(30,0,0,2);
    WaterSurfaceReplayState replay;
    std::vector<double> published {7,11};
    const auto priorPublished = published;
    check(!replay.advance(error) && !error.empty() && replay.height().empty() && replay.velocity().empty(),
        "unprepared water step refuses without a derived surface");
    check(!replay.publishHeight(published,error) && published==priorPublished,
        "unprepared water publication retains the previous surface");
    check(replay.prepare(history,op,rest,grid,3*dt,error) && replay.prepared() && !replay.complete()
        && replay.stepCount()==3 && replay.requestedTicks()==3*kParticleReplayStepTicks
        && replay.completedTicks()==0 && fixture::zero(replay.height()) && fixture::zero(replay.velocity())
        && replay.grid().columns==3 && replay.grid().rows==3,
        "one water preparation admits the full endpoint and starts with independent rest buffers");
    check(!replay.publishHeight(published,error) && published==priorPublished,
        "unfinished prepared water cannot publish a partial surface");
    check(replay.advance(error) && replay.completedTicks()==kParticleReplayStepTicks
        && fixture::zero(replay.height()) && fixture::zero(replay.velocity()),
        "first prepared interval excludes the audio impulse exactly on its end boundary");
    check(replay.advance(error) && replay.height().size()==9 && replay.velocity().size()==9,
        "prepared water advances the first completed impulse interval once");
    if (replay.height().size()!=9 || replay.velocity().size()!=9) return;
    check(fixture::near(replay.height()[4],2*dt) && replay.velocity()[4]==2
        && fixture::near(fixture::total(replay.height()),2*dt)
        && fixture::near(fixture::total(replay.velocity()),2),
        "prepared center height and velocity satisfy the independent impulse oracle");
    const auto firstHeight = replay.height(), firstVelocity = replay.velocity();
    check(!replay.publishHeight(published,error) && published==priorPublished
        && replay.height()==firstHeight && replay.velocity()==firstVelocity,
        "mid-replay publication refusal leaves the current admitted step intact");
    check(replay.advance(error) && replay.complete() && replay.completedTicks()==3*kParticleReplayStepTicks
        && fixture::near(replay.height()[4],2*dt*1.75) && fixture::near(replay.velocity()[4],1.5),
        "prepared propagation preserves the independent center stencil and exact 120 Hz clock");
    for (const auto i : {1,3,5,7})
        check(fixture::near(replay.height()[static_cast<std::size_t>(i)],2*dt/16)
            && fixture::near(replay.velocity()[static_cast<std::size_t>(i)],.125),
            "prepared propagation preserves the independent neighbor stencil");
    const auto completeHeight = replay.height(), completeVelocity = replay.velocity();
    check(!replay.advance(error) && !error.empty() && replay.height()==completeHeight
        && replay.velocity()==completeVelocity && replay.completedTicks()==3*kParticleReplayStepTicks,
        "a prepared water request cannot advance beyond its admitted completed intervals");
    check(replay.publishHeight(published,error) && published==completeHeight && !replay.prepared()
        && replay.height().empty() && replay.velocity().empty() && !replay.advance(error),
        "complete water publication moves the surface and consumes the request state");

    for (const auto seconds : {0.0,dt,1.75*dt,2*dt,2.75*dt,3*dt,.079,.6})
    {
        std::vector<double> ordinary;
        check(replayWaterSurface(history,op,rest,grid,seconds,ordinary,error)
            && replay.prepare(history,op,rest,grid,seconds,error),
            "ordinary and prepared water admit the same exact and fractional endpoint");
        bool advanced = true;
        while (advanced && !replay.complete()) advanced = replay.advance(error);
        check(advanced && replay.height().size()==ordinary.size()
            && std::memcmp(replay.height().data(),ordinary.data(),ordinary.size()*sizeof(double))==0,
            "stepwise water heights equal ordinary wrapper bytes at exact and fractional times");
    }
    // A prepared request owns the admitted impulse values, not live readers.
    std::vector<double> original;
    check(replayWaterSurface(history,op,rest,grid,3*dt,original,error)
        && replay.prepare(history,op,rest,grid,3*dt,error), "water prepares one owned input snapshot");
    history.score.reset(); history.audioFrames=nullptr; frames.clear();
    bool advanced = true;
    while (advanced && !replay.complete()) advanced = replay.advance(error);
    check(advanced && replay.height()==original,
        "prepared water steps do not reread mutated or released Score/audio history");
    const auto acceptedHeight = replay.height(), acceptedVelocity = replay.velocity();
    const auto acceptedTicks = replay.completedTicks();
    check(!replay.prepare(history,op,rest,grid,3*dt,error) && !error.empty()
        && replay.height()==acceptedHeight && replay.velocity()==acceptedVelocity
        && replay.completedTicks()==acceptedTicks && replay.complete(),
        "failed preparation cannot replace the previous admitted request with partial history");
    history=fixture::source(); frames=fixture::audio(1200); frames[0].onset=1; frames[0].onsetAge=0;
    history.audioFrames=&frames; history.audioSampleRate=61440;
    check(replay.prepare(history,op,rest,grid,3*dt,error) && replay.completedTicks()==0
        && fixture::zero(replay.height()) && fixture::zero(replay.velocity()),
        "the next admitted request after refusal starts from authored rest without old impulse cursors");
    while (!replay.complete()) if (!replay.advance(error)) break;
    check(replay.complete() && replay.height()==original,
        "refusal followed by valid admission reconstructs the same ordinary water result");

    // Score and audio share one interval. Independent total momentum checks
    // both causes; identity reordering checks the retained accumulation order.
    auto score=std::make_shared<arbitmod::Score>(*history.score);
    score->notes={fixture::note(11,.125f,30,63.5f),fixture::note(-7,.125f,60,127)};
    history.score=score; history.audioSampleRate=8192;
    frames=fixture::audio(2); frames[0].onset=1; frames[0].onsetAge=0;
    const auto mixed=fixture::operation(0,60,2,2);
    check(replay.prepare(history,mixed,rest,grid,9*dt,error),
        "prepared water admits simultaneous exact Score and master onset causes");
    while (!replay.complete()) if (!replay.advance(error)) break;
    check(replay.complete() && fixture::near(fixture::total(replay.velocity()),1.25)
        && fixture::near(fixture::total(replay.height()),3.75*dt),
        "prepared Score/audio total momentum matches the independent damped two-step oracle");
    const auto mixedHeight=replay.height(), mixedVelocity=replay.velocity();
    std::reverse(score->notes.begin(),score->notes.end());
    check(replay.prepare(history,mixed,rest,grid,9*dt,error), "prepared water admits reordered canonical rows");
    while (!replay.complete()) if (!replay.advance(error)) break;
    check(replay.complete() && replay.height()==mixedHeight && replay.velocity()==mixedVelocity,
        "prepared simultaneous impulses preserve tick-kind-identity ordering in both state buffers");

    history.audioSampleRate=61440; frames=fixture::audio(1200); frames[0].onset=1; frames[0].onsetAge=0;
    const auto uniformRest=fixture::mesh(2,2);
    WaterSurfaceGrid uniformGrid;
    const auto uniform=fixture::operation(60,0,0,2);
    check(validateWaterSurfaceGrid(uniformRest,uniformGrid,error)
        && replay.prepare(history,uniform,uniformRest,uniformGrid,5*dt,error),
        "prepared uniform grid admits the existing zero-gradient edge oracle");
    while (!replay.complete()) if (!replay.advance(error)) break;
    check(replay.complete(), "prepared uniform grid completes its bounded request");
    for (std::size_t i=0;i<replay.height().size();++i)
        check(fixture::near(replay.height()[i],2*dt) && fixture::near(replay.velocity()[i],.5),
            "prepared zero-gradient edges preserve uniform height and velocity without artificial boundary loss");

    const auto capped=fixture::mesh(32,32);
    WaterSurfaceGrid capGrid;
    const auto quiet=fixture::operation(30,0,0,0);
    check(validateWaterSurfaceGrid(capped,capGrid,error)
        && replay.prepare(history,quiet,capped,capGrid,10,error)
        && replay.stepCount()==1200 && replay.height().size()==1024 && replay.velocity().size()==1024,
        "prepared water preserves the ten-second and 32 by 32 admission bounds");
    while (!replay.complete()) if (!replay.advance(error)) break;
    check(replay.complete() && fixture::zero(replay.height()) && fixture::zero(replay.velocity()),
        "the maximal prepared quiet grid remains at rest through every bounded wave step");
}

// Independent center-stencil and total-momentum oracles exercise dynamics.
// These CPU fixtures do not establish editor, backend or export acceptance.
template <typename Check>
inline void waterSurfaceHistoryReplay(Check check)
{
    waterSurfacePreparedSteps(check);
    using namespace videorender;
    using namespace videowire::geometry;
    namespace fixture = waterreplayfixture;
    constexpr double dt = 1.0 / 120;
    std::string error;
    const auto rest = fixture::mesh(3, 3);
    WaterSurfaceGrid grid;
    check(validateWaterSurfaceGrid(rest, grid, error), "water replay fixture admits exact retained rest grid");
    auto history = fixture::source();
    auto op = fixture::operation(30, 0, 0, 2);
    auto frames = fixture::audio(1200);
    // 512/61440 = one exact 120 Hz step. The impulse starts interval 1.
    history.audioSampleRate = 61440; history.audioFrames = &frames;
    frames[0].onset = 1; frames[0].onsetAge = 0;
    for (const double mass:{.01,2.5,100.0}) {
        auto coupled=op; coupled.field.values[4]=mass;
        check(validateWaterSurfaceSettings(coupled,grid,error), "positive receiver mass is valid authored water settings");
        for (const double seconds:{0.0,3*dt}) {
            std::vector<double> unchanged{42,17};
            check(!replayWaterSurface(history,coupled,rest,grid,seconds,unchanged,error)
                && unchanged==std::vector<double>{42,17}
                && error.find("coupled frame coordinator")!=std::string::npos,
                "ordinary positive-mass replay refuses before publishing any heights, including project zero");
        }
    }
    std::vector<double> first, second, later, before, repeat;
    check(replayWaterSurface(history, op, rest, grid, 2 * dt, first, error), "master onset replays the first completed impulse interval");
    check(first.size() == 9 && fixture::near(first[4], 2 * dt)
          && fixture::near(fixture::total(first), 2 * dt), "center onset has the independently predicted total velocity impulse");
    if (first.size() != 9) return;
    for (std::size_t i = 0; i < first.size(); ++i)
        if (i != 4) check(first[i] == 0, "first center impulse leaves other vertices at rest");
    check(replayWaterSurface(history, op, rest, grid, 3 * dt, second, error), "water propagation advances a second fixed step");
    if (second.size() != 9) return;
    // c*dt=1/4, so the second update sends I*dt/16 to each neighbor.
    check(fixture::near(second[4], 2 * dt * 1.75), "independent five-point oracle predicts center after propagation");
    for (const auto i : {1, 3, 5, 7})
        check(fixture::near(second[static_cast<std::size_t>(i)], 2 * dt / 16), "independent five-point oracle predicts each immediate neighbor");
    for (const auto i : {0, 2, 6, 8})
        check(second[static_cast<std::size_t>(i)] == 0, "propagation has not reached diagonal vertices in the second step");
    check(fixture::near(fixture::total(second), 4 * dt), "reflective stencil preserves undamped total momentum");
    check(replayWaterSurface(history, op, rest, grid, dt, before, error) && fixture::zero(before), "onset at the completed boundary cannot affect the earlier interval");
    check(replayWaterSurface(history, op, rest, grid, 1.75 * dt, before, error) && fixture::zero(before), "onset in the unfinished interval cannot affect returned height");
    check(replayWaterSurface(history, op, rest, grid, 2.75 * dt, repeat, error) && repeat == first, "unfinished propagation interval contributes no variable frame delta");
    check(replayWaterSurface(history, op, rest, grid, 0.6, later, error)
          && replayWaterSurface(history, op, rest, grid, 2 * dt, repeat, error) && repeat == first,
          "seek order cannot change same-input water reconstruction");
    history.key.loopGeneration++; history.key.seekGeneration++;
    check(replayWaterSurface(history, op, rest, grid, 3 * dt, repeat, error) && repeat == second,
          "wrapped project time reproduces the same field across loop and seek generations");
    check(replayWaterSurface(history, op, rest, grid, 0, before, error) && fixture::zero(before), "project-zero reset restores authored rest");
    auto quiet = frames; quiet[0].onset = 0;
    history.audioFrames = &quiet;
    check(replayWaterSurface(history, op, rest, grid, 3 * dt, before, error) && fixture::zero(before), "master onset removal causally restores rest");
    history.audioFrames = &frames;
    auto softer = op; softer.field.values[3] = 1;
    check(replayWaterSurface(history, softer, rest, grid, 3 * dt, repeat, error), "audio gain control replays independently");
    for (std::size_t i = 0; i < second.size() && i < repeat.size(); ++i)
        check(fixture::near(repeat[i], second[i] / 2), "audio gain scales propagated height");
    auto damped = fixture::operation(0, 60, 0, 2);
    check(replayWaterSurface(history, damped, rest, grid, 2 * dt, first, error)
          && replayWaterSurface(history, damped, rest, grid, 3 * dt, second, error)
          && replayWaterSurface(history, damped, rest, grid, 4 * dt, later, error), "damped onset reconstructs successive fixed steps");
    check(fixture::near(fixture::total(first), dt)
          && fixture::near(fixture::total(second), 1.5 * dt)
          && fixture::near(fixture::total(later), 1.75 * dt), "independent geometric velocity-decay oracle matches damping");
    damped.field.values[1] = 120;
    check(replayWaterSurface(history, damped, rest, grid, 0.1, before, error) && fixture::zero(before), "full per-step damping safely zeros velocity");

    const auto uniformRest = fixture::mesh(2, 2);
    WaterSurfaceGrid uniformGrid;
    check(validateWaterSurfaceGrid(uniformRest, uniformGrid, error), "uniform boundary fixture admits");
    op = fixture::operation(60, 0, 0, 2);
    check(replayWaterSurface(history, op, uniformRest, uniformGrid, 5 * dt, before, error), "uniform impulse reconstructs through all reflective boundaries");
    for (const auto value : before) check(fixture::near(value, 2 * dt), "zero-gradient boundary leaves a uniform impulse spatially uniform");

    history = fixture::source();
    auto score = std::make_shared<arbitmod::Score>(*history.score);
    score->notes = {fixture::note(-7, 0, 0, 63.5f)};
    history.score = score;
    op = fixture::operation(0, 60, 2, 0);
    history.parameterAt = [](const auto&, double, double&, std::string&) { return false; };
    check(replayWaterSurface(history, op, rest, grid, dt, first, error)
          && first.size() == rest.positions.size() && fixture::near(fixture::total(first), 0.5 * dt),
          "canonical MIDI-unit velocity distributes one total score impulse without automation sampling");
    check(first[0] > 0 || first[3] > 0 || first[6] > 0, "pitch zero maps the stable note impulse to the grid's first X column");
    for (const auto i : {1, 2, 4, 5, 7, 8}) check(first[static_cast<std::size_t>(i)] == 0, "pitch-mapped impulse retains bilinear column support");
    check(replayWaterSurface(history, op, rest, grid, 2 * dt, second, error)
          && fixture::near(fixture::total(second), 0.75 * dt), "score velocity decay follows the independent closed form");
    score->notes[0].velocity = 127;
    check(replayWaterSurface(history, op, rest, grid, dt, repeat, error), "score velocity control replays");
    for (std::size_t i = 0; i < first.size(); ++i) check(fixture::near(repeat[i], 2 * first[i]), "MIDI velocity doubles score-driven height");
    score->notes[0].muted = true;
    check(replayWaterSurface(history, op, rest, grid, dt, before, error) && fixture::zero(before), "muted score notes do not inject impulses");
    score->notes[0].muted = false; score->notes[0].startBeat = -1;
    check(replayWaterSurface(history, op, rest, grid, dt, before, error) && fixture::zero(before), "pre-origin note-on does not become a fabricated zero-time impulse");
    score->notes = {fixture::note(-7, 0.025f)}; // 0.0125 seconds, inside interval 1.
    check(replayWaterSurface(history, op, rest, grid, 0.01, before, error) && fixture::zero(before), "future score note cannot retroactively affect earlier request");
    check(replayWaterSurface(history, op, rest, grid, 0.015, before, error) && fixture::zero(before), "score note in final unfinished interval remains unapplied");
    check(replayWaterSurface(history, op, rest, grid, 2 * dt, repeat, error) && fixture::total(repeat) > 0, "score impulse appears after its containing interval completes");
    // Exactly representable beat gives an event exactly at 0.0625 seconds.
    // A query just below it rounds to the same event tick but remains causal.
    score->notes = {fixture::note(-7, 0.125f)};
    check(replayWaterSurface(history, op, rest, grid, 0.0625 - 0.2 / kParticleReplayTicksPerSecond, before, error)
          && fixture::zero(before), "sub-tick rounding never exposes a future score impulse");
    score->notes = {fixture::note(-7, 0, 60, 1), fixture::note(11, 0, 30, 127), fixture::note(-12, 0, 100, 0.5f)};
    check(replayWaterSurface(history, op, rest, grid, 0.1, first, error), "simultaneous canonical identities replay");
    std::reverse(score->notes.begin(), score->notes.end());
    check(replayWaterSurface(history, op, rest, grid, 0.1, repeat, error) && repeat == first, "score row reordering retains exact sorted tick-kind-identity accumulation");

    const auto published = repeat;
    const auto rejected = [&](const ParticleHistorySource& candidate, const OperationRecord& settings,
                              const GeometryData& mesh, const WaterSurfaceGrid& shape, double seconds, const char* message)
    {
        auto output = published;
        const auto ok = replayWaterSurface(candidate, settings, mesh, shape, seconds, output, error);
        check(!ok && !error.empty() && output == published, message);
        const auto ordinaryDiagnostic=error;
        WaterSurfaceReplayState prepared;
        check(!prepared.prepare(candidate,settings,mesh,shape,seconds,error)
            && error==ordinaryDiagnostic && !prepared.prepared()
            && prepared.height().empty() && prepared.velocity().empty(),
            "prepared water admission preserves ordinary missing-history, nonfinite, topology and cap refusals");
    };
    auto badHistory = history; badHistory.score.reset();
    rejected(badHistory, op, rest, grid, dt, "missing canonical score refuses atomically");
    badHistory = history; badHistory.key.helperGeneration = 0;
    rejected(badHistory, op, rest, grid, dt, "invalid canonical source generation refuses atomically");
    auto invalidScore = std::make_shared<arbitmod::Score>(*score); invalidScore->notes.push_back(invalidScore->notes.front());
    badHistory = history; badHistory.score = invalidScore;
    rejected(badHistory, op, rest, grid, dt, "duplicate stable note identity refuses atomically");
    invalidScore = std::make_shared<arbitmod::Score>(*score); invalidScore->notes.back().startBeat = 100;
    invalidScore->notes.back().velocity = std::numeric_limits<float>::quiet_NaN(); badHistory.score = invalidScore;
    rejected(badHistory, op, rest, grid, dt, "whole canonical score audit rejects malformed future note outside replay window");
    auto overScore = std::make_shared<arbitmod::Score>(*history.score); overScore->notes.clear();
    for (int i = 1; i <= 512; ++i) overScore->notes.push_back(fixture::note(i, 0));
    badHistory = history; badHistory.score = overScore;
    check(replayWaterSurface(badHistory, op, rest, grid, dt, before, error), "exact 512 score impulse cap replays without truncation");
    overScore->notes.push_back(fixture::note(513, 0));
    rejected(badHistory, op, rest, grid, dt, "513 score impulses refuse atomically before stepping");
    op.field.values[2] = 0;
    check(replayWaterSurface(badHistory, op, rest, grid, dt, before, error) && fixture::zero(before), "disabled score gain does not request unnecessary impulse work");
    op.field.values[3] = 1;
    rejected(history, op, rest, grid, dt, "positive audio gain refuses missing master history before stepping");
    badHistory = history; badHistory.audioFrames = &frames; badHistory.audioSampleRate = 61440;
    auto malformed = frames; malformed[0].sampleIndex = 0; badHistory.audioFrames = &malformed;
    rejected(badHistory, op, rest, grid, 2 * dt, "shifted master hop origin refuses atomically");
    malformed = frames; malformed[1].sampleIndex += arbitblockb::kHop;
    rejected(badHistory, op, rest, grid, 2 * dt, "noncontiguous master hop coverage refuses atomically");
    malformed = frames; malformed[1].frameIndex = 2;
    rejected(badHistory, op, rest, grid, 2 * dt, "wrong master frame identity refuses atomically");
    malformed = frames; malformed[1].bands[17] = std::numeric_limits<float>::quiet_NaN();
    rejected(badHistory, op, rest, grid, 2 * dt, "nonfinite Block B frame refuses before stepping");
    malformed = frames; malformed[1].onsetAge = -1;
    rejected(badHistory, op, rest, grid, 2 * dt, "malformed onset age refuses before stepping");
    malformed = frames; malformed.resize(1);
    rejected(badHistory, op, rest, grid, 2 * dt, "missing final audio hop refuses rather than clamping endpoint history");
    malformed.clear();
    rejected(badHistory, op, rest, grid, 0, "positive gain requires master history even at reset");
    badHistory.audioFrames = &frames; badHistory.audioSampleRate = std::numeric_limits<double>::infinity();
    rejected(badHistory, op, rest, grid, dt, "nonfinite audio sample rate refuses atomically");
    badHistory.audioSampleRate = 0;
    rejected(badHistory, op, rest, grid, dt, "nonpositive audio sample rate refuses atomically");
    for (const auto hopCount : {1u, 3u, 9u, 17u})
    {
        auto exactCoverage = fixture::audio(hopCount);
        badHistory.audioFrames = &exactCoverage; badHistory.audioSampleRate = 44100;
        check(replayWaterSurface(badHistory, op, rest, grid, double(hopCount) * arbitblockb::kHop / 44100, before, error),
              "exact final master hop covers its endpoint without an extra rounded frame");
    }
    auto dense = fixture::audio(514);
    for (std::size_t i = 0; i < 513; ++i) { dense[i].onset = 1; dense[i].onsetAge = 0; }
    badHistory.audioFrames = &dense; badHistory.audioSampleRate = 61440;
    rejected(badHistory, op, rest, grid, 514 * dt, "513 audio impulses refuse rather than truncating history");
    auto combinedScore = std::make_shared<arbitmod::Score>(*history.score); combinedScore->notes.clear();
    for (int i = 1; i <= 512; ++i) combinedScore->notes.push_back(fixture::note(i, 0));
    badHistory.score = combinedScore; badHistory.audioFrames = &frames; op.field.values[2] = 1;
    rejected(badHistory, op, rest, grid, 2 * dt, "score and master onset share the same 512 impulse budget");
    op = fixture::operation(30, 0, 0, 0);
    check(replayWaterSurface(history, op, rest, grid, 10, before, error) && fixture::zero(before), "exact replay horizon returns rest with disabled sources");
    rejected(history, op, rest, grid, 10.001, "time beyond ten seconds refuses without silently dropping older waves");
    rejected(history, op, rest, grid, -dt, "negative project time refuses atomically");
    rejected(history, op, rest, grid, std::numeric_limits<double>::quiet_NaN(), "nonfinite project time refuses atomically");
    auto invalid = op; invalid.field.values[0] = 100;
    rejected(history, invalid, rest, grid, dt, "excess Courant coefficient refuses atomically");
    invalid = op; invalid.field.values[1] = 121;
    rejected(history, invalid, rest, grid, dt, "excess per-step damping refuses atomically");
    invalid = op; invalid.field.values[2] = std::numeric_limits<double>::infinity();
    rejected(history, invalid, rest, grid, dt, "nonfinite authored impulse gain refuses atomically");
    invalid = op; invalid.field.values[3] = 1001;
    rejected(history, invalid, rest, grid, dt, "over-cap authored impulse gain refuses atomically");
    auto forgedGrid = grid; forgedGrid.spacingX = 2;
    rejected(history, op, rest, forgedGrid, dt, "forged grid metadata refuses atomically");
    auto broken = rest; broken.vertexIds.back() = broken.vertexIds.front();
    rejected(history, op, broken, grid, dt, "duplicate rest vertex identities refuse atomically");
    broken = rest; broken.indices.back() = 0;
    rejected(history, op, broken, grid, dt, "malformed rest topology refuses atomically");
    const auto capped = fixture::mesh(32, 32);
    WaterSurfaceGrid capGrid;
    check(validateWaterSurfaceGrid(capped, capGrid, error)
          && replayWaterSurface(history, op, capped, capGrid, 10, before, error)
          && before.size() == 1024 && fixture::zero(before), "exact 32 by 32 vertex cap replays the full supported horizon");
    const auto oversized = fixture::mesh(33, 2);
    rejected(history, op, oversized, grid, dt, "over-cap water axis refuses atomically");
    check(std::all_of(repeat.begin(), repeat.end(), [](double value) { return std::isfinite(value); }), "published water offsets remain finite");
}


template <typename Check>
inline void waterSurfaceFrameCallback(Check check)
{
    using namespace videorender;
    using namespace videowire::geometry;
    namespace fixture = waterreplayfixture;
    std::string error;
    auto history = fixture::source();
    auto frames = fixture::audio(1200);
    frames[0].onset = 1; frames[0].onsetAge = 0;
    history.audioFrames = &frames; history.audioSampleRate = 61440;
    int parameterCalls = 0;
    history.parameterAt = [&](const auto&, double, double&, std::string&) { ++parameterCalls; return false; };
    const auto rest = fixture::mesh(3,3);
    const auto op = fixture::operation(30,0,0,2);
    WaterSurfaceGrid grid;
    check(validateWaterSurfaceGrid(rest,grid,error), "frame callback fixture retains exact grid");
    constexpr double projectSeconds = 3.0/120;
    constexpr double sceneSeconds = 23.5;
    auto currentFrame = fixture::frame(history,projectSeconds);
    auto evaluation = waterSurfaceFrameEvaluation(&history,projectSeconds,sceneSeconds,currentFrame);
    std::vector<double> expected, actual;
    check(replayWaterSurface(history,op,rest,grid,projectSeconds,expected,error)
        && evaluation.waterHeightAt(op,rest,grid,sceneSeconds,actual,error) && actual == expected,
        "frame callback anchors exact scene seconds to project time without a second offset");
    check(evaluation.waterHeightAt(op,rest,grid,sceneSeconds + 0.125,actual,error)
        && replayWaterSurface(history,op,rest,grid,projectSeconds + 0.125,expected,error) && actual == expected,
        "frame callback maps a scene delta relative to the exact supplied anchor");
    PortContract contract; contract.carrier = CarrierKind::geometry3D;
    contract.maxVertices = 32; contract.maxIndices = 128;
    auto authoredGrid = lowerGrid(11101,11101,1,3,3,1,contract,error);
    auto water = authoredGrid ? lowerWaterSurfaceSimulation(*authoredGrid,11102,11103,30,0,0,2,contract,error) : std::nullopt;
    auto value = water ? admitValue(*water,contract,error) : std::nullopt;
    check(value.has_value(), "frame memo fixture admits authored water rest");
    if (value) {
        using namespace HarmonicMIDI::grid;
        auto target = std::make_shared<Visual3DScene>(); target->id={7}; target->activeCamera={50};
        target->cameras[0].id={50}; target->cameraCount=1; target->cameras[0].transform.translation.z=4;
        SceneObjectRecord object; object.id={10}; SceneMaterialRecord material; material.id={20};
        check(appendGeometryToScene(*value,object,material,*target,error), "frame memo fixture appends authored rest");
        TimelineSceneBinding binding{{10},encodeLoweredPlanText(lowerRuntimePlan(contract,*value))};
        int replayCalls = 0;
        auto counted = evaluation;
        const auto producer = counted.waterHeightAt;
        counted.waterHeightAt = [&](const auto& settings,const auto& mesh,const auto& shape,double seconds,auto& heights,std::string& diagnostic) {
            ++replayCalls; return producer(settings,mesh,shape,seconds,heights,diagnostic);
        };
        check(replaceTimelineSceneMesh(binding,*value,sceneSeconds,*target,error,&counted) && replayCalls==1,
            "existing replacement memo calls the real water replay once per operation per frame");
    }
    // Production Preview/Export callback refuses the opt-in receiver until
    // the coupled coordinator owns the frame, without publishing mesh or normals.
    for (const bool exporting:{false,true}) {
        const auto ownedFrame=fixture::frame(history,projectSeconds,120,exporting);
        auto refusedEvaluation=waterSurfaceFrameEvaluation(&history,projectSeconds,sceneSeconds,ownedFrame);
        auto coupled=op; coupled.field.values[4]=2.5;
        std::vector<double> unchanged{42,17};
        check(!refusedEvaluation.waterHeightAt(coupled,rest,grid,sceneSeconds,unchanged,error)
            && unchanged==std::vector<double>{42,17}, "native Preview/Export callback cannot ignore positive receiver mass");
        auto authored=authoredGrid ? lowerWaterSurfaceSimulation(*authoredGrid,11102,11103,30,0,0,2,contract,error,2.5) : std::nullopt;
        const auto admitted=authored ? admitValue(*authored,contract,error):std::nullopt;
        check(bool(admitted), "positive-mass rest mesh admits before native frame evaluation");
        if (!admitted) continue;
        using namespace HarmonicMIDI::grid;
        auto target=std::make_shared<Visual3DScene>(); target->id={7}; target->activeCamera={50};
        target->cameras[0].id={50}; target->cameraCount=1; target->cameras[0].transform.translation.z=4;
        SceneObjectRecord object; object.id={10}; SceneMaterialRecord material; material.id={20};
        check(appendGeometryToScene(*admitted,object,material,*target,error), "positive-mass authored rest appends before replay");
        const auto before=sceneaov::detail::encodeScene(*target);
        TimelineSceneBinding binding{{10},encodeLoweredPlanText(lowerRuntimePlan(contract,*admitted))};
        check(!replaceTimelineSceneMesh(binding,*admitted,sceneSeconds,*target,error,&refusedEvaluation)
            && sceneaov::detail::encodeScene(*target)==before
            && error.find("coupled frame coordinator")!=std::string::npos,
            "refused positive-mass native frame preserves the entire published scene");
    }
    // Neither value grid lands on this exact nine-step replay request.
    constexpr double nonGridSeconds = 0.079;
    for (const double fps : {24.0,30000.0/1001.0}) {
        const auto previewFrame = fixture::frame(history,nonGridSeconds,fps);
        const auto exportFrame = fixture::frame(history,nonGridSeconds,fps,true);
        check(canonicalblockc::valid(previewFrame) && canonicalblockc::valid(exportFrame),
            "non-grid callback uses production Preview/Export FrameOwner packing");
        if (!previewFrame || !exportFrame) continue;
        const double snappedSeconds = static_cast<double>(previewFrame->key.frame) / fps;
        check(previewFrame->key.beat != static_cast<double>(static_cast<float>(history.timeline.secondsToBeat(nonGridSeconds))),
            "non-grid canonical beat differs from the exact replay beat");
        auto previewEvaluation = waterSurfaceFrameEvaluation(&history,nonGridSeconds,sceneSeconds,previewFrame);
        auto exportEvaluation = waterSurfaceFrameEvaluation(&history,nonGridSeconds,sceneSeconds,exportFrame);
        std::vector<double> direct,previewHeights,exportHeights,snappedHeights;
        check(replayWaterSurface(history,op,rest,grid,nonGridSeconds,direct,error)
            && previewEvaluation.waterHeightAt(op,rest,grid,sceneSeconds,previewHeights,error)
            && exportEvaluation.waterHeightAt(op,rest,grid,sceneSeconds,exportHeights,error)
            && previewHeights.size()==direct.size() && exportHeights.size()==direct.size()
            && std::memcmp(previewHeights.data(),direct.data(),direct.size()*sizeof(double))==0
            && std::memcmp(exportHeights.data(),direct.data(),direct.size()*sizeof(double))==0,
            "non-grid preview/export heights equal independent exact-time replay bytes");
        check(replayWaterSurface(history,op,rest,grid,snappedSeconds,snappedHeights,error) && snappedHeights!=direct,
            "canonical beat quantization does not snap the 120 Hz water replay endpoint");
        const auto rejectFrame = [&](const auto& candidate,const ParticleHistorySource& source,const char* message) {
            auto refused = waterSurfaceFrameEvaluation(&source,nonGridSeconds,sceneSeconds,candidate);
            auto result = direct;
            check(!refused.waterHeightAt(op,rest,grid,sceneSeconds,result,error)
                && !error.empty() && result==direct,message);
        };
        auto foreign = history; ++foreign.key.seekGeneration;
        rejectFrame(previewFrame,foreign,"non-grid callback still refuses foreign lifecycle without publishing heights");
        rejectFrame(fixture::frame(history,nonGridSeconds+1/fps,fps),history,
            "non-grid callback still refuses another canonical frame without publishing heights");
        auto wrongTimeline = history; wrongTimeline.timeline.reset(90,4);
        rejectFrame(fixture::frame(wrongTimeline,nonGridSeconds,fps),history,
            "non-grid callback still refuses a different packed beat without publishing heights");
    }
    const auto staticOperation = fixture::operation(30,0,0,0);
    auto staticEvaluation = waterSurfaceFrameEvaluation(&history,nonGridSeconds,sceneSeconds,
        fixture::frame(history,nonGridSeconds,24));
    check(history.score->notes.empty() && history.score->scoreRevision!=0
        && staticEvaluation.waterHeightAt(staticOperation,rest,grid,sceneSeconds,actual,error) && fixture::zero(actual),
        "owned published empty score supplies the ordinary static water frame");
    auto absentScore = history; absentScore.score = std::make_shared<const arbitmod::Score>();
    auto absentFrame = fixture::frame(absentScore,nonGridSeconds,24);
    auto absentEvaluation = waterSurfaceFrameEvaluation(&absentScore,nonGridSeconds,sceneSeconds,absentFrame);
    auto unchanged = actual;
    check(!absentFrame && !absentEvaluation.waterHeightAt(staticOperation,rest,grid,sceneSeconds,unchanged,error)
        && !error.empty() && unchanged==actual,
        "revision-zero default score cannot invent a canonical frame for static water");
    check(parameterCalls == 0, "static water controls never query parameter history");
    const auto published = actual;
    const auto reject = [&](const ParticleHistorySource* source,double project,double scene,double requested,const char* message)
    {
        auto candidate = waterSurfaceFrameEvaluation(source,project,scene,currentFrame);
        auto result = published;
        check(!candidate.waterHeightAt(op,rest,grid,requested,result,error) && !error.empty() && result == published,message);
    };
    reject(nullptr,projectSeconds,sceneSeconds,sceneSeconds,"missing frame-owned history refuses without publishing heights");
    auto invalid = history; invalid.key.projectGeneration = 0;
    reject(&invalid,projectSeconds,sceneSeconds,sceneSeconds,"invalid frame history generation refuses without publishing heights");
    invalid = history; invalid.audioFrames = nullptr;
    reject(&invalid,projectSeconds,sceneSeconds,sceneSeconds,"midrange export without project-zero audio refuses without publishing heights");
    reject(&history,10.001,sceneSeconds,sceneSeconds,"callback preserves the exact ten-second replay horizon");
    reject(&history,projectSeconds,sceneSeconds,sceneSeconds - 1,"scene delta cannot invent a negative project origin");
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    reject(&history,nan,sceneSeconds,sceneSeconds,"callback refuses a nonfinite project anchor");
    reject(&history,projectSeconds,nan,sceneSeconds,"callback refuses a nonfinite scene anchor");
    reject(&history,projectSeconds,sceneSeconds,nan,"callback refuses a nonfinite scene request");
    history.key.seekGeneration++; history.key.loopGeneration++;
    evaluation = waterSurfaceFrameEvaluation(&history,projectSeconds,sceneSeconds,fixture::frame(history,projectSeconds));
    check(evaluation.waterHeightAt(op,rest,grid,sceneSeconds,actual,error)
        && replayWaterSurface(history,op,rest,grid,projectSeconds,expected,error) && actual == expected,
        "generation changes retain project-zero replay and do not fabricate a loop reset");
}
