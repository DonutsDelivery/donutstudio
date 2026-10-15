#pragma once

#include "water_surface_replay_fixture.h"
#include "../src/water_replay_source_budget.h"

template <typename Check>
inline void waterReplaySourceBudget(Check check)
{
    using namespace videorender;
    using namespace videowire::geometry;
    using namespace watersourcebudget;
    namespace fixture = waterreplayfixture;
    std::string error;
    auto history = fixture::source();
    auto score = std::make_shared<arbitmod::Score>(*history.score);
    // All 512 impulses are at zero. The other 1536 notes are future source
    // records, still validated and tempo-scanned despite Block C's 128 rows.
    for (int i = 1; i <= 2048; ++i)
    {
        auto note = fixture::note(i, i <= 512 ? 0.0f : 100.0f);
        note.pitchBendPoints.resize(3);
        for (int a = 1; a <= 4; ++a) note.pitchAnchors.push_back({a,0,440});
        score->notes.push_back(std::move(note));
    }
    for (int i = 1; i <= 300; ++i) score->links.push_back({i,i,i+1,1,1,0});
    history.score = score;
    std::vector<videotime::TempoPoint> tempos;
    for (int i = 0; i < 20; ++i) tempos.push_back({double(i),120,true});
    history.timeline.set(std::move(tempos), {{0,4,4},{8,3,4}});
    auto audio = fixture::audio(1200);
    history.audioFrames = &audio; history.audioSampleRate = 61440;
    const auto mesh = fixture::mesh(3,3);
    WaterSurfaceGrid grid;
    check(validateWaterSurfaceGrid(mesh,grid,error), "source-budget fixture has genuine regular Water geometry");
    auto operation = fixture::operation(30,0,1,1);
    constexpr double project = 1.0/120, local = 23.5;
    const auto frame = fixture::frame(history,project);
    check(bool(frame), "source-budget fixture has a real canonical frame for a score longer than resident rows");
    if (!frame) return;
    Recipe inspected;
    int consumers = 0;
    const auto inspect = [&](const auto& actualSource,const auto& actualOperation,const auto& actualMesh,
        const auto& actualGrid,const auto& actualFrame,double actualProject,double actualLocal,double actualRequested,
        const Recipe& recipe,std::string&) {
        ++consumers; inspected = recipe;
        check(&actualSource==&history && &actualOperation==&operation && &actualMesh==&mesh
            && &actualGrid==&grid && &actualFrame==&frame && actualProject==project
            && actualLocal==local && actualRequested==local,
            "preflight passes complete actual source objects and exact anchors synchronously");
        return true;
    };
    Limits limits;
    check(withBudget(history,operation,mesh,grid,frame,project,local,local,limits,inspect,error),
        "private prospective recipe admits named source allowances without evaluating Water");
    // Expected counts are independently derived from the authored fixture.
    check(inspected.sourceNotes==2048 && inspected.sourceLinks==300 && inspected.sourceAudioFrames==1200
        && inspected.sourceTempoPoints==20 && inspected.sourceMeterPoints==2 && inspected.audioPrefix==1
        && inspected.sourceRestPositions==9 && inspected.sourceRestVertexIds==9 && inspected.sourceRestIndices==24
        && inspected.steps==1 && inspected.impulseCapacity==512,
        "recipe retains complete source sizes and the real coverage prefix separately from event cap");
    check(inspected.at(Work::censusNotes)==2048 && inspected.at(Work::scoreNotes)==2048
        && inspected.at(Work::scoreMasterNotes)==2048 && inspected.at(Work::scoreLinks)==300
        && inspected.at(Work::primes)==12288 && inspected.at(Work::commas)==18432
        && inspected.at(Work::bends)==6144 && inspected.at(Work::anchors)==8192
        && inspected.at(Work::priorAnchorComparisons)==12288 && inspected.at(Work::scoreIdentityCalls)==4996,
        "all nested score visits and quadratic anchor comparisons exceed resident row limits honestly");
    check(inspected.at(Work::beatToSecondsCalls)==2048 && inspected.at(Work::secondsToBeatCalls)==1
        && inspected.at(Work::tempoSegmentVisits)==38931 && inspected.at(Work::tempoPointReads)==79911
        && inspected.at(Work::scoreOnsetScan)==2048 && inspected.at(Work::impulseCandidates)==2049,
        "tempo full-scan upper bounds include future notes with the same 512 accepted impulses");
    check(inspected.at(Work::audioValidationFrames)==1 && inspected.at(Work::audioValidationScalars)==5
        && inspected.at(Work::audioValidationBands)==64 && inspected.at(Work::audioOnsetScan)==1
        && inspected.at(Work::canonicalTextureValues)==3072 && inspected.at(Work::canonicalNoteRows)==128
        && inspected.at(Work::canonicalLinkRows)==256 && inspected.at(Work::gridColumnScan)==9
        && inspected.at(Work::gridVertices)==9 && inspected.at(Work::gridIndices)==24
        && inspected.at(Work::gridIdentityCalls)==9 && inspected.at(Work::settingsValues)==12,
        "recipe separates canonical, grid, settings and master audio validation work");
    check(inspected.at(Work::sortElements)==512 && inspected.at(Work::waveInitializationElements)==36
        && inspected.at(Work::velocityCopyElements)==9 && inspected.at(Work::eventCursorChecks)==513
        && inspected.at(Work::eventDeposits)==2048 && inspected.at(Work::stencilVertices)==9
        && inspected.at(Work::stencilHeightReads)==63 && inspected.at(Work::publicationTransfers)==1
        && inspected.bytes(Storage::waveArrays)==288
        && inspected.bytes(Storage::impulses)==512*sizeof(waterreplay::Impulse)
        && inspected.bytes(Storage::noteIdentities)==2048*sizeof(int)
        && inspected.bytes(Storage::linkIdentities)==300*sizeof(int)
        && inspected.bytes(Storage::gridIdentities)==9*sizeof(StableId),
        "fresh replay preparation, event/stencil loops and move publication have separate logical payloads");

    // Keep numerical output from the released evaluator independently. The
    // prospective recipe above is not evidence that these predicates executed.
    std::vector<double> direct, callback;
    auto released = waterSurfaceFrameEvaluation(&history,project,local,frame);
    check(replayWaterSurface(history,operation,mesh,grid,project,direct,error)
        && released.waterHeightAt(operation,mesh,grid,local,callback,error) && direct==callback
        && direct.size()==9 && std::all_of(direct.begin(),direct.end(),[](double v) { return std::isfinite(v); }),
        "independent released numerical replay remains available for the later budget join");
    auto shortScore = std::make_shared<arbitmod::Score>(*score); shortScore->notes.resize(512);
    auto shortHistory = history; shortHistory.score = shortScore;
    std::vector<double> shortOutput;
    check(replayWaterSurface(shortHistory,operation,mesh,grid,project,shortOutput,error) && shortOutput==direct
        && withBudget(shortHistory,operation,mesh,grid,frame,project,local,local,limits,
            [&](const auto&,const auto&,const auto&,const auto&,const auto&,double,double,double,const Recipe& recipe,std::string&) {
                return recipe.impulseCapacity==512 && recipe.at(Work::scoreOnsetScan)==512
                    && recipe.at(Work::priorAnchorComparisons)==3072 && recipe.at(Work::tempoSegmentVisits)==9747;
            },error),"same 512 real impulses and identical released heights still require distinct complete source scan budgets");
    const auto refuseBudget = [&](const Limits& shortLimits,const char* message) {
        const auto prior = consumers;
        check(!withBudget(history,operation,mesh,grid,frame,project,local,local,shortLimits,inspect,error)
            && consumers==prior && !error.empty(),message);
    };
    auto shortLimits = limits; shortLimits.work[static_cast<std::size_t>(Work::censusNotes)]=2047;
    refuseBudget(shortLimits,"short census refuses before nested note-size traversal or consumer work");
    for (std::size_t i = 1; i < inspected.work.size(); ++i)
    {
        if (inspected.work[i]==0) continue;
        shortLimits = limits; shortLimits.work[i]=inspected.work[i]-1;
        refuseBudget(shortLimits,"each named work shortage refuses before source validations and replay");
    }
    for (std::size_t i = 0; i < inspected.logicalBytes.size(); ++i)
    {
        shortLimits = limits; shortLimits.logicalBytes[i]=inspected.logicalBytes[i]-1;
        refuseBudget(shortLimits,"each logical payload shortage refuses before storage allocation");
    }
    // Coverage mirrors the real floor/end-point rule, including exact hops.
    auto audioOnly = fixture::operation(30,0,0,1);
    for (const auto hops : {1u,3u,9u,17u})
    {
        auto endpointSource = history; endpointSource.audioSampleRate = 44100;
        endpointSource.audioFrames = &audio;
        const double seconds = double(hops)*arbitblockb::kHop/44100;
        check(withBudget(endpointSource,audioOnly,mesh,grid,frame,seconds,local,local,limits,
            [&](const auto&,const auto&,const auto&,const auto&,const auto&,double,double,double,const Recipe& recipe,std::string&) {
                return recipe.audioPrefix==hops;
            },error),"exact-hop budget coverage uses the released endpoint rounding rule");
    }
    const auto reject = [&](const ParticleHistorySource& source,const OperationRecord& op,double seconds,double scene,double query,
        const char* message) {
        bool reached = false;
        check(!withBudget(source,op,mesh,grid,frame,seconds,scene,query,limits,
            [&](const auto&...){ reached=true; return true; },error) && !reached && !error.empty(),message);
    };
    auto extreme = history; extreme.audioSampleRate = std::numeric_limits<double>::max();
    reject(extreme,audioOnly,10,local,local,"real source with overflowing rate/time product refuses before conversion or audio traversal");
    extreme.audioSampleRate = 1e25;
    reject(extreme,audioOnly,project,local,local,"real source with unrepresentable prefix refuses without integer conversion");
    extreme.audioSampleRate = 61440*1201.0;
    reject(extreme,audioOnly,project,local,local,"extreme finite rate requires complete actual coverage rather than clamping source size");
    reject(history,operation,10.001,local,local,"over-horizon query refuses before source consumer");
    reject(history,operation,project,local,local-1,"negative project query refuses");
    reject(history,operation,project,std::numeric_limits<double>::max(),-std::numeric_limits<double>::max(),
        "finite anchors with overflowing delta refuse");
    auto coupled=operation; coupled.field.values[4]=1;
    reject(history,coupled,project,local,local,"uncoupled source recipe cannot grant coupled frame reuse");
    std::uint64_t result=7;
    check(!checkedMultiply(std::numeric_limits<std::uint64_t>::max(),2,result) && result==7
        && !videorender::watersourcebudget::checkedAdd(std::numeric_limits<std::uint64_t>::max(),1,result)
        && !checkedPairs(std::numeric_limits<std::uint64_t>::max(),result),
        "synthetic arithmetic boundaries refuse independently of feasible source allocation");

    // Named counts cannot authenticate substituted identities/settings/time.
    // The later consumer must run the released predicates against these inputs.
    auto foreign = history; ++foreign.key.seekGeneration;
    auto changed = operation; changed.field.values[0]=20;
    const auto substitution = [&](const ParticleHistorySource& source,const OperationRecord& op,double seconds) {
        return withBudget(source,op,mesh,grid,frame,seconds,local,local,limits,
            [&](const auto& actual,const auto& settings,const auto&,const auto&,const auto& current,double global,double,double,
                const Recipe& recipe,std::string& diagnostic) {
                check(&actual==&source && &settings==&op && &current==&frame
                    && recipe.sourceNotes==2048,"recipe carries actual substituted sources without a digest grant");
                diagnostic="fixture consumer rejects substituted request";
                return &actual==&history && &settings==&operation && global==project;
            },error);
    };
    check(!substitution(foreign,operation,project) && !substitution(history,changed,project)
        && !substitution(history,operation,2*project),
        "source, operation and time substitutions reach their actual consumer and receive no budget authentication");
    const auto otherFrame = fixture::frame(history,2*project);
    const auto rejectReleased = [&](const ParticleHistorySource& source,const auto& current,double seconds) {
        auto output = direct;
        const auto admitted = withBudget(source,operation,mesh,grid,current,seconds,local,local,limits,
            [&](const auto& actual,const auto& settings,const auto& rest,const auto& shape,const auto& canonical,
                double global,double scene,double requested,const Recipe&,std::string& diagnostic) {
                auto evaluator = waterSurfaceFrameEvaluation(&actual,global,scene,canonical);
                return evaluator.waterHeightAt(settings,rest,shape,requested,output,diagnostic);
            },error);
        return !admitted && !error.empty() && output==direct;
    };
    check(rejectReleased(foreign,frame,project) && rejectReleased(history,otherFrame,project)
        && rejectReleased(history,frame,2*project),
        "later real released consumer still rejects source lifecycle, current-frame and global-time substitutions atomically");
}
