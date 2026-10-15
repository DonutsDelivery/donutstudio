#pragma once

#include "particle_parameters.h"
#include "particle_history.h"
#include "particle_geometry.h"
#include "particle_solid_state.h"

namespace videorender
{
// Births follow the existing replay: a body appears when its sounding note or
// emitter is first admitted, or after a replay reset. Note end and window end
// bound its life. These times describe the same bodies drawn by the Image path.
struct ParticlePointReplayTimes {
    std::array<double,kParticleSolidMaxBodies> birthSeconds {}, lifetimeSeconds {};
    double sampleSeconds = 0;
};

// Request-local simulation authority for coupled particles on every backend.
// History audio and the optional input frame are borrowed. Prepare, drain and
// publish synchronously while their source owner lives; retain no preview state.
// A prior-water view is borrowed only within each prepare/advance call. Supply
// the same frozen view for the whole call; only its cache identity is retained.
class ParticleBodyReplayState
{
public:
    ParticleBodyReplayState()=default;
    ParticleBodyReplayState(const ParticleBodyReplayState&)=delete;
    ParticleBodyReplayState& operator=(const ParticleBodyReplayState&)=delete;
    bool prepare(const ParticleParams& parameters, double seconds, float ratio,
        const canonicalblockc::CanonicalBlockCFrame* sourceFrame, std::string& error,
        bool needSolidState = false, bool needPointTimes = false,
        const ParticlePriorWaterGeometryView* priorWater = nullptr,
        const std::shared_ptr<const ParticleBodyCallerLease>& callerGeometryResources = {})
    {
        const bool authenticated=requiresAuthenticatedParticleBodyReplay(parameters);
        std::shared_ptr<const ParticleBodyCallerLease> scratch,retained,publication;
        if (authenticated) {
            const auto& context=parameters.bodyReplayContext;
            if (!context || !context->admitted()) {
                error_="Dynamic body replay requires its admitted stable caller context";
                prepared_=false; failed_=true; error=error_; return false;
            }
            scratch=context->caller->reserve(callerScratchLogicalBytes(),callerPrepareWork());
            retained=scratch ? context->caller->reserve(ParticleBodyReplayReceipt::retainedLogicalBytes(),callerReceiptWork()):nullptr;
            const auto outputBytes=callerOutputLogicalBytes(needSolidState,needPointTimes);
            publication=retained ? context->caller->reserve(outputBytes,outputBytes):nullptr;
            if (!scratch || !retained || !publication) {
                error_="Dynamic body replay prospective caller admission is short";
                prepared_=false; failed_=true; error=error_; return false;
            }
        }
        // Stage under the new reservation before retiring an old request. Its
        // ordinary destructor releases payloads before its first-declared lease.
        auto staged=parameters;
        this->~ParticleBodyReplayState();
        new (this) ParticleBodyReplayState{};
        callerScratch_=std::move(scratch); callerRetained_=std::move(retained); callerOutput_=std::move(publication);
        input=std::move(staged); p=input; clipSeconds=seconds; aspect=ratio; frame=sourceFrame;
        wantSolidState=needSolidState; wantPointTimes=needPointTimes;
        for (auto& value : result) value={-10,-10,0,0};
        prepared_=prepareRequest(priorWater,callerGeometryResources); failed_=!prepared_;
        if (!prepared_ && error_.empty()) error_="Particle body replay preparation failed without a diagnostic";
        error=error_; return prepared_;
    }

    static constexpr std::size_t callerScratchLogicalBytes() noexcept
    {
        // State includes request/working Params, bodies, collider, boundary array,
        // producer and publication scratch. Extra locals and old/candidate
        // canonical frames coexist during history and wrapper publication.
        return sizeof(ParticleBodyReplayState)+3*sizeof(ParticleParams)
            +sizeof(ParticleBodyState)+sizeof(ParticleSolidState)+sizeof(ParticlePointReplayTimes)
            +3*sizeof(canonicalblockc::CanonicalBlockCFrame);
    }
    static constexpr std::size_t callerPrepareWork() noexcept
    { return 2*sizeof(ParticleBodyReplayState)+3*sizeof(ParticleParams); }
    static constexpr std::size_t callerReceiptWork() noexcept
    { return 2*(sizeof(ParticleBodyReplayReceipt)+sizeof(ParticleBodyState)+sizeof(ParticleSolidState)); }
    static constexpr std::size_t callerScheduleCopyWork(std::size_t boundaries) noexcept
    {
        // sampleHistory is called at the initial tick, each interval start and
        // the terminal tick. An initial tick may be read twice but sampled once.
        return (boundaries+1)*sizeof(ParticleParams);
    }
    // Each interval may write all 128 captures 8,192, reset 1,024 guard bytes and
    // write the 16-byte descriptor and 8-byte generation. Each capacity body adds
    // 27,768 workspace bytes plus 32 view construction and 32 retirement bytes.
    static constexpr std::size_t callerRotatingCopyWork(std::size_t steps,std::size_t bodies) noexcept
    { return steps*(9240+bodies*27832); }
    static constexpr std::size_t callerOutputLogicalBytes(bool solid,bool times) noexcept
    { return sizeof(ParticleBodyState)+(solid ? sizeof(ParticleSolidState):0)+(times ? sizeof(ParticlePointReplayTimes):0); }
    // The fresh synchronous wrapper keeps all three reservations through
    // receipt issuance and publication. Each reserve adds its own lease header.
    static constexpr std::size_t callerFreshPeakLogicalBytes(bool solid,bool times) noexcept
    {
        return callerScratchLogicalBytes()+ParticleBodyReplayReceipt::retainedLogicalBytes()
            +callerOutputLogicalBytes(solid,times)+3*sizeof(ParticleBodyCallerLease);
    }
    bool prepared() const { return prepared_; }
    bool complete() const { return prepared_ && !failed_ && completed_; }
    const std::shared_ptr<const ParticleBodyReplayReceipt>& receipt() const noexcept { return receipt_; }
    const std::shared_ptr<const ParticleBodyGeometrySampler>& geometrySampler() const noexcept { return bodySampler; }

    // Only a successfully completed historical subinterval has absolute ticks.
    // These are the existing reset/cycle origin plus its exact split boundaries.
    bool completedIntervalTicks(std::int64_t& start, std::int64_t& end) const
    {
        if (!prepared_ || failed_ || !p.historicalReplay || !hasInterval_) return false;
        start=intervalStart_; end=intervalEnd_; return true;
    }

    bool nextIntervalTicks(std::int64_t& start, std::int64_t& end) const
    {
        if (!prepared_ || failed_ || completed_ || !p.historicalReplay) return false;
        start=startTicks+boundaries[static_cast<std::size_t>(step)];
        end=startTicks+boundaries[static_cast<std::size_t>(step+1)]; return true;
    }

    std::function<bool(const SolidTriangleContactImpulse&, const ParticleGeometryFrame&, std::string&)> waterContact;
    std::uint64_t waterMeshIdentity=0;

    bool advanceOne(std::string& error, const ParticlePriorWaterGeometryView* priorWater = nullptr)
    {
        if (!prepared_ || failed_) {
            error=error_.empty() ? "Particle step requires a prepared replay request":error_;
            return false;
        }
        if (completed_) { error="Particle step exceeds the prepared replay boundary"; return false; }
        if (!advanceInterval(priorWater)) {
            // A silent false here would publish the fill value as an empty body set.
            failed_=true;
            if (error_.empty()) error_="Particle body replay interval failed without a diagnostic";
            error=error_; return false;
        }
        if (p.historicalReplay) {
            intervalStart_=startTicks+boundaries[static_cast<std::size_t>(step)];
            intervalEnd_=startTicks+boundaries[static_cast<std::size_t>(step+1)];
            hasInterval_=true;
        }
        ++step;
        if (step==steps && !finish(priorWater)) { failed_=true; error=error_; return false; }
        error.clear(); return true;
    }

    // Refusal leaves all caller outputs unchanged. The scalar wrapper starts
    // its outputs empty and only publishes the completed request.
    bool publish(ParticleBodyState& output, std::string& error,
        ParticleSolidState* solidState = nullptr, ParticlePointReplayTimes* pointTimes = nullptr) const
    {
        if (!complete()) {
            error=error_.empty() ? "Particle publication requires a complete prepared replay":error_;
            return false;
        }
        if ((solidState && !wantSolidState) || (pointTimes && !wantPointTimes)) {
            error="Particle publication outputs were not requested during preparation"; return false;
        }
        // Reserve each publication separately; an escaped solid output retains
        // its own lease instead of refreshing storage on the next publication.
        std::shared_ptr<const ParticleBodyCallerLease> outputLease;
        if (dynamicBody) {
            const auto bytes=callerOutputLogicalBytes(solidState!=nullptr,pointTimes!=nullptr);
            outputLease=callerOutput_ ? std::move(callerOutput_):callerScratch_->ledger->reserve(bytes,bytes);
            if (!outputLease) { error="Dynamic body publication prospective caller admission is short"; return false; }
        }
        output=result;
        if (solidState) { *solidState=publishedSolid; if (dynamicBody) solidState->callerLease=std::move(outputLease); }
        if (pointTimes) *pointTimes=publishedTimes;
        error.clear(); return true;
    }

private:
    // First members die last, after all retained replay and inline payloads.
    std::shared_ptr<const ParticleBodyCallerLease> callerScratch_,callerRetained_;
    mutable std::shared_ptr<const ParticleBodyCallerLease> callerOutput_;
    using Body=ParticleRigidBody;
    struct Link { int a, b; float rest, weight; };
    static constexpr auto ticksPerSecond=kParticleReplayTicksPerSecond, stepTicks=kParticleReplayStepTicks;
    bool prepared_=false, failed_=false, completed_=false, hasInterval_=false;
    bool wantSolidState=false, wantPointTimes=false, solid=false, onsetInCycle=false;
    ParticleParams input, p;
    double clipSeconds=0, elapsed=0, age=0, sourceSeconds=0, replayEndSeconds=0, emissionBudget=0;
    double previousParameterTime=0, impulseAt=0;
    float aspect=1, duration=0, mass=0, radius=0, previousReset=0, previousResetTime=0;
    const canonicalblockc::CanonicalBlockCFrame* frame=nullptr;
    ParticleReplayWindow window;
    std::int64_t ageTicks=0, startTicks=0, lastOnset=-1, intervalStart_=0, intervalEnd_=0;
    SolidVector boxHalf {};
    std::array<Body,kParticleSolidMaxBodies> bodies {};
    std::array<Link,arbitblockc::kMaxLinks> links {};
    int count=0, linkCount=0, capacity=0, boundaryCount=0, steps=0, step=0;
    std::array<double,kParticleSolidMaxBodies> births {}, lifetimes {};
    canonicalblockc::FrameProducer historicalProducer;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> historicalFrame;
    ParticleHistoryAudio historicalAudio;
    ParticleGeometryFrame geometryFrame;
    SolidTriangleWorkBudget triangleWork;
    // Inline rotating-plane storage, 9,272 + 27,768 bytes, so the sizeof-based
    // caller scratch and prepare work include it once. rotatingLimbs is the
    // preflight-admitted exact width, or zero when the class is unavailable,
    // which includes every replay without an authenticated caller lease.
    SolidRotationLedger rotations;
    rotatingplane::Workspace rotatingWorkspace;
    unsigned rotatingLimbs=0;
    std::int64_t geometryTick=std::numeric_limits<std::int64_t>::min();
    std::uint64_t geometryWaterIdentity=0;
    std::int64_t geometryWaterStart=0, geometryWaterEnd=0;
    std::array<std::int64_t,1201+kParticleReplayMaxEvents> boundaries {};
    // Visited score events that carry an adapter weight, sorted by window-relative
    // tick. Each one is also a boundary, so an interval that starts at its tick
    // sees it. The attractor span is evaluated on the boundary grid.
    struct ScoreAdapterEvent { std::int64_t tick = 0; ParticleScoreEvent event {}; };
    std::array<ScoreAdapterEvent,kParticleReplayMaxEvents> scoreEvents {};
    int scoreEventCount=0;
    // Score attractors pull toward this height, the middle of the planar frame.
    static constexpr float kScoreAttractorY=0.5f;
    ParticleBodyState result {};
    ParticleSolidState publishedSolid;
    ParticlePointReplayTimes publishedTimes;
    std::string error_;
    bool dynamicBody=false;
    std::shared_ptr<const ParticleBodyGeometrySampler> bodySampler;
    std::shared_ptr<const ParticleBodyGeometrySample> bodySample;
    std::shared_ptr<const ParticleBodyReplayReceipt> receipt_;

    bool prepareRequest(const ParticlePriorWaterGeometryView* priorWater,
        const std::shared_ptr<const ParticleBodyCallerLease>& callerGeometryResources)
    {
        auto* diagnostic=&error_;
        dynamicBody=requiresAuthenticatedParticleBodyReplay(input);
        if (dynamicBody) {
            const auto& context=input.bodyReplayContext;
            if (!context || !context->owner.valid() || !input.history || !input.history->score
                || input.motionMode!=2 || !input.historicalReplay || input.simulationSpace!=1
                || !input.geometryBinding || !input.geometryBinding->body || !input.geometryBinding->bodySource
                || input.historyNodeId<=0 || input.historyClipId<=0
                || context->owner.clipIdentity!=static_cast<std::uint64_t>(input.historyClipId)
                || context->owner.planGeneration!=input.geometryRevision
                || context->owner.projectGeneration!=input.history->key.projectGeneration
                || context->owner.helperGeneration!=input.history->key.helperGeneration
                || context->owner.deviceGeneration!=input.history->key.deviceGeneration
                || !context->caller || !context->caller->onCallerThread()
                || context->width<=0 || context->height<=0
                || aspect!=float(context->width)/context->height || context->frame.get()!=frame
                || !std::isfinite(clipSeconds) || !std::isfinite(input.historyProjectSeconds)
                || std::abs(clipSeconds)>9007199254740991.0/ticksPerSecond
                || input.historyProjectSeconds<0 || input.historyProjectSeconds>9007199254740991.0/ticksPerSecond
                || priorWater) {
                error_="Dynamic body replay requires its exact direct owner, source, clocks, frame and extent"; return false;
            }
            if ((frame && (!(frame->key==input.history->key) || !canonicalblockc::valid(context->frame)))
                || (!input.history->score->notes.empty() && !frame)) {
                error_="Dynamic body replay frame differs from its full history frame key"; return false;
            }
            if (input.preparedBodies || input.preparedSolids || input.preparedBodyReplay) {
                if (!input.preparedBodyReplay || !input.preparedBodyReplay->matchesPrepared(input,clipSeconds,aspect,frame)) {
                    error_="Dynamic body prepared replay does not match its completed request"; return false;
                }
                if (wantPointTimes) {
                    error_="Completed direct body replay does not retain requested point times"; return false;
                }
                receipt_=input.preparedBodyReplay;
            }
        }
        if (priorWater && (!p.historicalReplay || p.motionMode!=2)) {
            error_="Particle prior-water view requires historical Coupled body replay"; return false;
        }
        if (!priorWater && !wantPointTimes && input.preparedBodies && (!wantSolidState || input.preparedSolids)) {
            if (wantSolidState) publishedSolid=*input.preparedSolids;
            result=*input.preparedBodies; completed_=true; return true;
        }
        if (p.historicalReplay && p.history && p.history->parameterAt && p.historyNodeId>0)
        {
            double resetTime=p.resetTime;
            std::string error;
            if (!p.history->parameterAt("clip"+std::to_string(p.historyClipId)+"/visual"
                +std::to_string(p.historyNodeId)+"/resetTime",p.historyProjectSeconds,resetTime,error)
                || !std::isfinite(resetTime))
            { if (diagnostic) *diagnostic=error.empty() ? "Automated reset time is nonfinite":error; return false; }
            p.resetTime=std::clamp(static_cast<float>(resetTime),0.0f,86400.0f);
        }
        elapsed = clipSeconds - p.resetTime;
        if (!std::isfinite(elapsed) || elapsed < 0) {
            if (dynamicBody && !issueReceipt()) return false;
            completed_=true; return true;
        }
        duration = std::clamp(p.lifetime, 0.1f, 10.0f);
        // The shared rational clock represents 120 Hz steps and common 44.1/48 kHz
        // audio sample rates exactly, including their 96/192 kHz multiples.
        window = particleReplayWindow(p, clipSeconds);
        ageTicks = window.age;
        age = p.historicalReplay ? double(ageTicks) / ticksPerSecond
            : (p.resetMode == 0 ? std::fmod(elapsed, duration) : std::min(elapsed, double(duration)));
        startTicks = window.start;
        solid=p.simulationSpace==1;
        if (solid && p.bodyShape==2 && !dynamicBody && !solidBodyBoxHalfExtents(p,boxHalf)) {
            if (diagnostic) *diagnostic="Oriented box requires static body geometry with three nonzero dimensions";
            return false;
        }
        capacity = std::clamp(p.count, 1, particleMaxBodies(p.simulationSpace));
        mass = std::clamp(p.bodyMass, 0.01f, 100.0f);
        sourceSeconds = p.historicalReplay ? double(startTicks)/ticksPerSecond : p.resetTime;
        replayEndSeconds = sourceSeconds + duration;
        previousResetTime=p.resetTime;
        previousParameterTime=double(startTicks)/ticksPerSecond-(input.historyProjectSeconds-clipSeconds);
        if (p.historicalReplay)
        {
            boundaries[static_cast<std::size_t>(boundaryCount++)] = 0;
            for (std::int64_t tick = stepTicks; tick < ageTicks; tick += stepTicks)
                boundaries[static_cast<std::size_t>(boundaryCount++)] = tick;
            if (!visitParticleScoreEvents(p, window, [&](std::int64_t tick, const ParticleScoreEvent& event)
                {
                    boundaries[static_cast<std::size_t>(boundaryCount++)] = tick;
                    if (event.kick > 0 || event.emission > 0 || event.attractor > 0)
                        scoreEvents[static_cast<std::size_t>(scoreEventCount++)] = {tick, event};
                }, p.scoreImpulseGain, p.scoreEmitGain, p.scoreAttractGain)) {
                if (diagnostic) *diagnostic=particleEventCapDiagnostic(p,window,"Coupled replay exceeds 512 note/audio event boundaries");
                return false;
            }
            boundaries[static_cast<std::size_t>(boundaryCount++)] = ageTicks;
            std::sort(boundaries.begin(), boundaries.begin() + boundaryCount);
            boundaryCount = static_cast<int>(std::unique(boundaries.begin(), boundaries.begin() + boundaryCount) - boundaries.begin());
            std::sort(scoreEvents.begin(), scoreEvents.begin() + scoreEventCount,
                [](const ScoreAdapterEvent& a, const ScoreAdapterEvent& b)
                {
                    if (a.tick != b.tick) return a.tick < b.tick;
                    if (a.event.kick != b.event.kick) return a.event.kick < b.event.kick;
                    if (a.event.emission != b.event.emission) return a.event.emission < b.event.emission;
                    return a.event.attractor < b.event.attractor;
                });
        }
        steps = p.historicalReplay ? boundaryCount - 1
            : std::min(1200, static_cast<int>(std::ceil(age * 120)));
        if (dynamicBody) {
            if (startTicks<0 || ageTicks<0 || startTicks>9007199254740991LL-ageTicks) {
                error_="Dynamic body replay requires representable absolute schedule ticks"; return false;
            }
            if (!input.bodyReplayContext->caller->charge(callerScheduleCopyWork(static_cast<std::size_t>(boundaryCount)))) {
                error_="Dynamic body replay prospective parameter-copy work is short"; return false;
            }
            ParticleBodyGeometryAttempt admissionAttempt;
            const auto admitted=admitParticleBodyGeometry(input.geometryBinding->bodySource,
                p.bodyShape==2 ? ParticleBodyGeometryProxy::fixedBox:ParticleBodyGeometryProxy::sphere,
                static_cast<std::size_t>(boundaryCount),input.bodyReplayContext->admissionLimits,admissionAttempt,error_,callerGeometryResources);
            ParticleBodyGeometrySampleAttempt sampleAttempt;
            bodySampler=admitted ? makeParticleBodyGeometrySampler(admitted,input.bodyReplayContext->sampleLimits,sampleAttempt,error_,callerGeometryResources):nullptr;
            if (!bodySampler) return false;
        }
        if (p.historicalReplay) {
            if (!sampleHistory(startTicks,false,priorWater)) return false;
        } else updateInputs(frame,p.rms,false);
        radius=std::clamp(p.bodyRadius,0.001f,0.1f);
        impulseAt=age-p.onsetAge;
        onsetInCycle=p.onsetAge<=elapsed && impulseAt>=0;
        auto preflightWork=triangleWork;
        if (priorWater) {
            // Each new water endpoint pair invalidates the cached start sample.
            // Reserve those extra validations before running either simulation.
            const auto samples=std::uint64_t(steps+1)*(geometryFrame.meshVertexCount+geometryFrame.meshTriangleCount);
            if (samples>preflightWork.limit) { error_="Water/body geometry sample work exceeds its bound"; return false; }
            preflightWork.limit-=samples;
        }
        if (solid && geometryFrame.meshIdentity
            && !preflightWork.preflight(static_cast<std::uint64_t>(steps),capacity,
                geometryFrame.meshVertexCount,geometryFrame.meshTriangleCount,geometryFrame.pointCount,p.bodyShape==2 ? 96:1)) {
            if (diagnostic) *diagnostic="Triangle surface replay exceeds 100M worst-case checked work units; reduce replay work";
            return false;
        }
        // Admit the rotating box class before any step: the largest exact width
        // whose whole prospective work fits beside the fixed-axis bound, then
        // each interval's capture/guard/view/workspace copy charge. Captures
        // are request-local through the caller lease, so only an authenticated
        // dynamic replay can admit a width. A nonzero width is the only
        // condition for every capture write, and it implies this charge.
        if (dynamicBody && solid && p.bodyShape==2 && geometryFrame.meshIdentity)
            rotatingLimbs=preflightWork.rotatingWidth(static_cast<std::uint64_t>(steps),capacity,
                geometryFrame.meshVertexCount,geometryFrame.meshTriangleCount,geometryFrame.pointCount);
        if (rotatingLimbs
            && !input.bodyReplayContext->caller->charge(callerRotatingCopyWork(static_cast<std::size_t>(steps),
                static_cast<std::size_t>(capacity)))) {
            error_="Dynamic body replay prospective rotating capture copy work is short"; return false;
        }
        lastOnset = p.historicalReplay ? historicalAudio.onsetSample : -1;
        if (p.historicalReplay && lastOnset >= 0 && p.history->hasAudio()
            && std::llround(double(lastOnset) / p.history->audioSampleRate * ticksPerSecond) >= startTicks)
            lastOnset = -1; // An onset exactly on reset excites the new bodies once.
        return steps==0 ? finish(priorWater) : true;
    }

    void updateInputs(const canonicalblockc::CanonicalBlockCFrame* sourceFrame, float rms, bool preserve)
    {
        this->frame = sourceFrame;
        const auto* frame = sourceFrame;
        const auto previous = bodies;
        const int previousCount = count;
        const auto previousBirths = births, previousLifetimes = lifetimes;
        count = 0; linkCount = 0;
        std::array<int, arbitblockc::kMaxNotes> rows {};
        int rowCount = 0;
        if (frame != nullptr)
        {
            for (int row = 0; row < std::clamp(frame->noteRows(), 0, arbitblockc::kMaxNotes); ++row)
            {
                const auto at = static_cast<std::size_t>(row * 16);
                const auto& values = frame->noteTextureValues;
                if (std::isfinite(values[at]) && std::isfinite(values[at + 1])
                    && values[at + 1] > 0.001f && values[at + 2] >= 0 && values[at + 3] > 0
                    && values[at + 6] == static_cast<float>(p.spawnTrack)
                    && frame->noteIdentities[static_cast<std::size_t>(row)] != 0)
                    rows[static_cast<std::size_t>(rowCount++)] = row;
            }
            std::sort(rows.begin(), rows.begin() + rowCount, [frame](int a, int b)
            { return frame->noteIdentities[static_cast<std::size_t>(a)]
                   < frame->noteIdentities[static_cast<std::size_t>(b)]; });
        }
        const auto hash = [](uint32_t n)
        {
            n = (n << 13U) ^ n;
            n = n * (n * n * 15731U + 789221U) + 1376312589U;
            return float(n & 0x7fffffffU) / float(0x7fffffff);
        };
        const int sources = p.geometryCount > 0 ? std::clamp(p.geometryCount, 0, 64) : rowCount;
        for (int index = 0; index < std::min(capacity, sources); ++index)
        {
            const int row = p.geometryCount > 0 ? -1 : rows[static_cast<std::size_t>(index)];
            const float pitch = row >= 0 ? frame->noteTextureValues[static_cast<std::size_t>(row * 16)] : 60;
            const float velocity = row >= 0 ? std::clamp(frame->noteTextureValues[static_cast<std::size_t>(row * 16 + 1)], 0.0f, 1.0f) : 1;
            const float angle = (hash(static_cast<uint32_t>(index) * 31U + static_cast<uint32_t>(p.seed)) - 0.5f) * 2.2f;
            const float speed = (0.25f + velocity * 0.75f) * p.force * (1 + p.rmsGain * rms);
            Body body {};
            body.x = std::clamp((pitch - 36) / 60, 0.0f, 1.0f) * 0.8f + 0.1f;
            body.y = 0.12f; body.tx = body.x; body.ty = body.y;
            body.z=body.tz=solid ? p.spawnZ:0;
            if (p.geometryCount > 0)
            {
                const auto& point = p.geometryAnchors[static_cast<std::size_t>(index)];
                body.x = point[0]; body.y = point[1]; body.tx = point[2]; body.ty = point[3];
                body.geometryIdentity=p.geometryIdentities[static_cast<std::size_t>(index)];
                if (solid) { body.z=p.geometryDepths[index][0]; body.tz=p.geometryDepths[index][1]; }
            }
            body.vx = std::clamp(std::sin(angle) * speed / (solid ? 1.0f:std::max(aspect, 0.001f)) + p.impulseX * velocity / mass, -32.0f, 32.0f);
            body.vy = std::clamp(std::cos(angle) * speed + p.impulseY * velocity / mass, -32.0f, 32.0f);
            if (solid) {
                body.vz=std::clamp(p.impulseZ*velocity/mass,-32.0f,32.0f);
                const float factor=0.00872664626f;
                const float sx=std::sin(p.orientationDegrees[0]*factor),cx=std::cos(p.orientationDegrees[0]*factor);
                const float sy=std::sin(p.orientationDegrees[1]*factor),cy=std::cos(p.orientationDegrees[1]*factor);
                const float sz=std::sin(p.orientationDegrees[2]*factor),cz=std::cos(p.orientationDegrees[2]*factor);
                body.orientation={sx*cy*cz-cx*sy*sz,cx*sy*cz+sx*cy*sz,cx*cy*sz-sx*sy*cz,cx*cy*cz+sx*sy*sz};
                body.angular=p.angularVelocity;
            }
            body.hue = pitch / 12 - std::floor(pitch / 12);
            body.noteRow = row;
            body.identity = row >= 0 ? frame->noteIdentities[static_cast<std::size_t>(row)] : index;
            double birth=sourceSeconds;
            double life=std::max(0.0,replayEndSeconds-birth);
            if (wantPointTimes && row >= 0 && p.history && p.history->score) {
                const auto* note=p.history->score->noteById(static_cast<int>(body.identity));
                if (note) life=std::max(0.0,std::min(replayEndSeconds,p.history->timeline.beatToSeconds(note->endBeat()))-birth);
            }
            bool existed=false;
            if (preserve)
                for (int old = 0; old < previousCount; ++old)
                    if (previous[static_cast<std::size_t>(old)].identity == body.identity
                        && previous[static_cast<std::size_t>(old)].geometryIdentity == body.geometryIdentity)
                    {
                        const auto& saved = previous[static_cast<std::size_t>(old)];
                        body.x = saved.x; body.y = saved.y; body.vx = saved.vx; body.vy = saved.vy;
                        body.z=saved.z; body.vz=saved.vz; body.orientation=saved.orientation; body.angular=saved.angular;
                        birth=previousBirths[static_cast<std::size_t>(old)];
                        life=previousLifetimes[static_cast<std::size_t>(old)];
                        existed=true;
                        break;
                    }
            if (!existed && p.emissionRate>=0)
            {
                if (emissionBudget<1) continue;
                emissionBudget-=1;
            }
            births[static_cast<std::size_t>(count)]=birth;
            lifetimes[static_cast<std::size_t>(count)]=life;
            bodies[static_cast<std::size_t>(count++)] = body;
        }
        if (frame != nullptr && p.geometryCount == 0 && p.linkSpring > 0)
        {
            std::array<int, arbitblockc::kMaxLinks> order {};
            const int rowsInLinks = std::clamp(frame->linkRows(), 0, arbitblockc::kMaxLinks);
            for (int i = 0; i < rowsInLinks; ++i) order[static_cast<std::size_t>(i)] = i;
            std::sort(order.begin(), order.begin() + rowsInLinks, [frame](int a, int b)
            { return frame->linkIdentities[static_cast<std::size_t>(a)] < frame->linkIdentities[static_cast<std::size_t>(b)]; });
            for (int i = 0; i < rowsInLinks; ++i)
            {
                const int row = order[static_cast<std::size_t>(i)];
                if (frame->linkIdentities[static_cast<std::size_t>(row)] == 0) continue;
                const auto at = static_cast<std::size_t>(row * 4);
                const auto& values = frame->linkTextureValues;
                int a = -1, b = -1;
                for (int j = 0; j < count; ++j)
                {
                    if (values[at] == float(bodies[static_cast<std::size_t>(j)].noteRow)) a = j;
                    if (values[at + 1] == float(bodies[static_cast<std::size_t>(j)].noteRow)) b = j;
                }
                if (a < 0 || b < 0 || a == b || !std::isfinite(values[at + 2])
                    || !std::isfinite(values[at + 3]) || values[at + 2] <= 0 || values[at + 3] <= 0) continue;
                const float ratio = std::clamp(std::max(values[at + 2] / values[at + 3], values[at + 3] / values[at + 2]), 1.0f, 8.0f);
                const auto& ba = bodies[static_cast<std::size_t>(a)];
                const auto& bb = bodies[static_cast<std::size_t>(b)];
                links[static_cast<std::size_t>(linkCount++)] = { a, b,
                    std::max(0.001f, (solid ? std::hypot(std::hypot(bb.tx-ba.tx,bb.ty-ba.ty),bb.tz-ba.tz)
                        : std::hypot(bb.tx-ba.tx,bb.ty-ba.ty)) * p.linkRestScale),
                    1 + p.linkRatioInfluence * (ratio - 1) };
            }
        }
    }

    bool sampleGeometry(std::int64_t tick, const ParticlePriorWaterGeometryView* priorWater)
    {
        auto* diagnostic=&error_;
        if (priorWater && !priorWater->covers(tick,tick,error_)) return false;
        const auto waterIdentity=priorWater ? priorWater->identity:0;
        const auto waterStart=priorWater ? priorWater->intervalStartTicks:0;
        const auto waterEnd=priorWater ? priorWater->intervalEndTicks:0;
        if (tick==geometryTick && waterIdentity==geometryWaterIdentity
            && waterStart==geometryWaterStart && waterEnd==geometryWaterEnd) return true;
        std::string error;
        if (!sampleParticleGeometry(p,tick,geometryFrame,error,priorWater)
            || (solid && !validateSolidTriangleFrame(geometryFrame,triangleWork,error)))
        { if (diagnostic) *diagnostic=error; return false; }
        geometryTick=tick; geometryWaterIdentity=waterIdentity;
        geometryWaterStart=waterStart; geometryWaterEnd=waterEnd;
        return true;
    }

    // creditOnsets is set only by the fixed-step interval, so the window-start
    // sample that prepare takes before step 0 never credits the same onset twice.
    bool sampleHistory(std::int64_t tick, bool preserve, const ParticlePriorWaterGeometryView* priorWater,
        bool creditOnsets=false)
    {
        auto* diagnostic=&error_;
        if (!p.history || !p.history->score) {
            if (diagnostic) *diagnostic="Coupled replay requires the owned project score history";
            return false;
        }
        const visualdeformation::RationalFrameTime time {tick, static_cast<std::uint32_t>(ticksPerSecond), 1};
        const double seconds = double(time.frame) * time.rateDenominator / time.rateNumerator;
        std::string parameterError;
        if (!sampleParticleParameters(input,p,seconds,parameterError))
        { if (diagnostic) *diagnostic=parameterError; return false; }
        if (priorWater && (p.resetTime!=0 || p.reset!=0))
        { error_="Water body coupling does not support mid-history particle resets"; return false; }
        mass=std::clamp(p.bodyMass,.01f,100.0f);
        const double localTime=seconds-(input.historyProjectSeconds-clipSeconds);
        const bool resetEdge=(p.reset>=.5f && previousReset<.5f)
            || (localTime>=p.resetTime && (p.resetTime!=previousResetTime || previousParameterTime<p.resetTime));
        previousReset=p.reset; previousResetTime=p.resetTime;
        previousParameterTime=localTime;
        if (resetEdge) { preserve=false; emissionBudget=0; }
        historicalAudio = p.history->audioAt(seconds);
        historicalFrame.reset();
        if (seconds >= 0 && !p.history->score->notes.empty())
        {
            auto key = p.history->key;
            key.frame = time.frame; key.fps = time.rateNumerator;
            const float beat = static_cast<float>(p.history->timeline.secondsToBeat(seconds));
            key.beat = beat;
            historicalFrame = historicalProducer.evaluate(key, p.history->score, beat);
            if (!historicalFrame) {
                if (diagnostic) *diagnostic="Coupled replay could not evaluate the canonical score at its fixed-step time";
                return false;
            }
        }
        if (dynamicBody) {
            if (!bodySample || bodySample->absoluteTick()!=tick) {
                ParticleBodyGeometrySampleAttempt attempt;
                auto candidate=bodySampler->sample(input.geometryBinding->bodySource,
                    input.geometryBinding->bodySource->descriptor().sourceRevision,tick,attempt,error_);
                if (!candidate) return false;
                bodySample=std::move(candidate);
            }
            if (p.bodyShape==2) {
                const auto& ratios=bodySample->boxHalfRatios();
                for (std::size_t axis=0;axis<3;++axis) boxHalf[axis]=p.bodyRadius*ratios[axis];
                if (!std::all_of(boxHalf.begin(),boxHalf.end(),[](float v) { return std::isfinite(v) && v>1.0e-6f; })) {
                    error_="Dynamic box body requires positive authored contact half extents"; return false;
                }
            }
        }
        if (!sampleGeometry(tick,priorWater)) return false;
        sourceSeconds=seconds;
        // Emission credit from onsets at this interval's start. It lands before
        // updateInputs, so an onset's own births can spend it at that tick.
        if (creditOnsets && p.emissionRate>=0)
        {
            double credit=0;
            for (int i=0;i<scoreEventCount;++i)
                if (scoreEvents[static_cast<std::size_t>(i)].tick==tick-startTicks)
                    credit+=scoreEvents[static_cast<std::size_t>(i)].event.emission;
            emissionBudget=std::min(64.0,emissionBudget+credit);
        }
        updateInputs(historicalFrame.get(), historicalAudio.rms, preserve);
        return true;
    }

    void contact(Body& a, Body& b)
    {
        const float dx = b.x - a.x, dy = b.y - a.y;
        float nx = 1, ny = 0, penetration = 0;
        if (p.bodyShape == 1)
        {
            const float px = 2 * radius - std::abs(dx), py = 2 * radius - std::abs(dy);
            if (px <= 0 || py <= 0) return;
            if (px <= py) { nx = dx < 0 ? -1.0f : 1.0f; penetration = px; }
            else { nx = 0; ny = dy < 0 ? -1.0f : 1.0f; penetration = py; }
        }
        else
        {
            const float distance = std::hypot(dx, dy);
            if (distance >= 2 * radius) return;
            if (distance > 1.0e-6f) { nx = dx / distance; ny = dy / distance; }
            penetration = 2 * radius - distance;
        }
        a.x -= nx * penetration * 0.5f; a.y -= ny * penetration * 0.5f;
        b.x += nx * penetration * 0.5f; b.y += ny * penetration * 0.5f;
        const float relative = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
        if (relative >= 0) return;
        const float impulse = -(1 + p.restitution) * relative * 0.5f;
        const float tangent = std::clamp(((b.vx - a.vx) * -ny + (b.vy - a.vy) * nx) * -0.5f,
                                         -p.friction * impulse, p.friction * impulse);
        a.vx -= nx * impulse - ny * tangent; a.vy -= ny * impulse + nx * tangent;
        b.vx += nx * impulse - ny * tangent; b.vy += ny * impulse + nx * tangent;
    }

    void walls(Body& body)
    {
        if (p.collisionMode == 0) return;
        if (body.x < radius) { body.x = radius; body.vx = std::abs(body.vx) * p.restitution; body.vy *= 1 - p.friction; }
        if (body.x > 1 - radius) { body.x = 1 - radius; body.vx = -std::abs(body.vx) * p.restitution; body.vy *= 1 - p.friction; }
        if (body.y < radius) { body.y = radius; body.vy = std::abs(body.vy) * p.restitution; body.vx *= 1 - p.friction; }
        if (body.y > 1 - radius) { body.y = 1 - radius; body.vy = -std::abs(body.vy) * p.restitution; body.vx *= 1 - p.friction; }
    }

    bool advanceInterval(const ParticlePriorWaterGeometryView* priorWater)
    {
        auto* diagnostic=&error_;
        if (priorWater && !priorWater->covers(startTicks+boundaries[static_cast<std::size_t>(step)],
            startTicks+boundaries[static_cast<std::size_t>(step+1)],error_)) return false;
        const double start = p.historicalReplay ? double(boundaries[static_cast<std::size_t>(step)]) / ticksPerSecond : double(step) / 120;
        const float dt = p.historicalReplay
            ? static_cast<float>(double(boundaries[static_cast<std::size_t>(step + 1)] - boundaries[static_cast<std::size_t>(step)]) / ticksPerSecond)
            : static_cast<float>(std::min(1.0 / 120, age - start));
        bool historicalImpulse = false;
        if (p.historicalReplay)
        {
            if (!sampleHistory(startTicks + boundaries[static_cast<std::size_t>(step)], true, priorWater, true)) return false;
            radius=std::clamp(p.bodyRadius,.001f,.1f);
            historicalImpulse = historicalAudio.onsetSample >= 0 && historicalAudio.onsetSample != lastOnset;
            lastOnset = historicalAudio.onsetSample;
        }
        // A score onset at this interval's start kicks every body in the interval.
        // Equal ticks sum. A note sounding over the whole interval attracts every
        // body toward its pitch point. Both depend only on the note timeline and step.
        const auto boundaryTick = boundaries[static_cast<std::size_t>(step)];
        double scoreKick = 0, attractStrength = 0, attractXWeighted = 0;
        if (p.historicalReplay)
            for (int i = 0; i < scoreEventCount; ++i)
            {
                const auto& entry = scoreEvents[static_cast<std::size_t>(i)];
                if (entry.tick == boundaryTick) scoreKick += entry.event.kick;
                if (entry.event.attractor > 0 && entry.tick <= boundaryTick && boundaryTick < entry.event.attractorEnd)
                {
                    attractStrength += entry.event.attractor;
                    attractXWeighted += entry.event.attractor * entry.event.attractorX;
                }
            }
        // Unit direction of the initial impulse controls, or +Y when all are zero.
        // Planar replay has no Z component, so it ignores impulseZ. hypot avoids
        // the squared-sum underflow that would send a tiny direction to +Y.
        std::array<float,3> kickDirection {0,1,0};
        if (scoreKick > 0)
        {
            const float dx=p.impulseX, dy=p.impulseY, dz=solid ? p.impulseZ : 0.0f;
            const float length=std::hypot(std::hypot(dx,dy),dz);
            if (length > 0) kickDirection={dx/length,dy/length,dz/length};
        }
        emissionBudget=std::min(64.0,emissionBudget+std::max(0.0,double(p.emissionRate))*dt);
        if (rotatingLimbs) rotations.begin(p.historicalReplay ? startTicks+boundaries[static_cast<std::size_t>(step)]:step,
            p.historicalReplay ? startTicks+boundaries[static_cast<std::size_t>(step+1)]:step+1);
        const auto previousGeometry=geometryFrame;
        const auto previousBodies=bodies;
        std::array<std::array<float, 3>, kParticleSolidMaxBodies> forces {};
        for (int index = 0; index < linkCount && p.linkConstraint == 0; ++index)
        {
            const auto& link = links[static_cast<std::size_t>(index)];
            const auto& a = bodies[static_cast<std::size_t>(link.a)];
            const auto& b = bodies[static_cast<std::size_t>(link.b)];
            const float dx = b.x - a.x, dy = b.y - a.y, dz=b.z-a.z;
            const float distance = solid ? std::hypot(std::hypot(dx,dy),dz):std::hypot(dx,dy);
            if (distance < 1.0e-6f) continue;
            const float force = std::clamp(p.linkSpring * link.weight * (distance - link.rest), -100.0f, 100.0f) / distance;
            forces[static_cast<std::size_t>(link.a)][0] += dx * force;
            forces[static_cast<std::size_t>(link.a)][1] += dy * force;
            forces[static_cast<std::size_t>(link.b)][0] -= dx * force;
            forces[static_cast<std::size_t>(link.b)][1] -= dy * force;
            forces[static_cast<std::size_t>(link.a)][2] += dz * force;
            forces[static_cast<std::size_t>(link.b)][2] -= dz * force;
        }
        for (int i = 0; i < count; ++i)
        {
            auto& body = bodies[static_cast<std::size_t>(i)];
            if (p.historicalReplay && historicalImpulse)
                body.vy += p.onsetGain * historicalAudio.onset / mass;
            if (scoreKick > 0)
            {
                const float impulse = static_cast<float>(scoreKick) / mass;
                body.vx += kickDirection[0] * impulse; body.vy += kickDirection[1] * impulse;
                if (solid) body.vz += kickDirection[2] * impulse;
            }
            if (!p.historicalReplay && onsetInCycle && ((step == 0 && impulseAt == 0) || (impulseAt > start && impulseAt <= start + dt)))
                body.vy += p.onsetGain * p.onset / mass;
            // Dense links and small masses share fixed numerical bounds, so a
            // hostile but admitted graph cannot grow an infinite trajectory.
            // Score attraction sums strength_j * (target_j - body) over sounding notes.
            const float ax = std::clamp((forces[static_cast<std::size_t>(i)][0] + p.attraction * (body.tx - body.x)
                + float(attractXWeighted - attractStrength * body.x)) / mass
                + (solid ? p.gravityX:0), -1000.0f, 1000.0f);
            const float ay = std::clamp((forces[static_cast<std::size_t>(i)][1] + p.attraction * (body.ty - body.y)
                + float(attractStrength * kScoreAttractorY - attractStrength * body.y)
                + (p.historicalReplay ? p.rmsGain * historicalAudio.rms : 0)) / mass - p.gravity, -1000.0f, 1000.0f);
            body.vx = std::clamp((body.vx + ax * dt) / (1 + p.drag * dt), -32.0f, 32.0f);
            body.vy = std::clamp((body.vy + ay * dt) / (1 + p.drag * dt), -32.0f, 32.0f);
            body.x += body.vx * dt; body.y += body.vy * dt;
            if (solid) {
                const float az=std::clamp((forces[i][2]+p.attraction*(body.tz-body.z)
                    + float(attractStrength*(p.spawnZ-body.z)))/mass+p.gravityZ,-1000.0f,1000.0f);
                body.vz=std::clamp((body.vz+az*dt)/(1+p.drag*dt),-32.0f,32.0f);
                body.z+=body.vz*dt;
                if (rotatingLimbs) rotations.rotate(body,static_cast<std::size_t>(i),dt,p.angularDrag);
                else solidRotate(body,dt,p.angularDrag);
            }
        }
        if (p.geometryBinding && !sampleGeometry(startTicks + boundaries[static_cast<std::size_t>(step + 1)],priorWater)) return false;
        const auto geometryContact = [&](Body& body, const Body& previous, int index)
        {
            const auto& edge=geometryFrame.segments[static_cast<std::size_t>(index)];
            const auto& old=previousGeometry.segments[static_cast<std::size_t>(index)];
            const float ex=edge[2]-edge[0],ey=edge[3]-edge[1],length2=ex*ex+ey*ey;
            const float t=length2>1.0e-12f ? std::clamp(((body.x-edge[0])*ex+(body.y-edge[1])*ey)/length2,0.0f,1.0f) : 0;
            const float cx=edge[0]+t*ex,cy=edge[1]+t*ey;
            float dx=body.x-cx,dy=body.y-cy,distance=std::hypot(dx,dy);
            const float reach=radius+p.colliderThickness;
            const float motionX=((edge[0]-old[0])+(edge[2]-old[2]))*0.5f;
            const float motionY=((edge[1]-old[1])+(edge[3]-old[3]))*0.5f;
            const float fromX=previous.x+motionX,fromY=previous.y+motionY;
            const float travelX=body.x-fromX,travelY=body.y-fromY;
            const float cross=travelX*ey-travelY*ex;
            bool crossed=false, endpointHit=false;
            float firstHit=2;
            float hitX=cx,hitY=cy;
            float pointNormalX=1,pointNormalY=0;
            if (std::abs(cross)>1.0e-10f)
            {
                const float qx=edge[0]-fromX,qy=edge[1]-fromY;
                const float along=(qx*ey-qy*ex)/cross,segment=(qx*travelY-qy*travelX)/cross;
                crossed=along>=0 && along<=1 && segment>=0 && segment<=1;
                if (crossed) { firstHit=along; hitX=fromX+along*travelX; hitY=fromY+along*travelY; }
            }
            // Sweep the round end caps too. A fast body can touch a mesh
            // vertex without crossing the segment's center line.
            for (int endpoint=0;endpoint<(length2<=1.0e-12f ? 1 : 2);++endpoint)
            {
                const float px=edge[endpoint*2],py=edge[endpoint*2+1];
                const float qx=fromX-px,qy=fromY-py,a=travelX*travelX+travelY*travelY;
                const float b=2*(qx*travelX+qy*travelY),c=qx*qx+qy*qy-reach*reach;
                const float discriminant=b*b-4*a*c;
                if (a>1.0e-12f && discriminant>=0 && c>=0)
                {
                    const float along=(-b-std::sqrt(discriminant))/(2*a);
                    if (along>=0 && along<=1 && along<firstHit)
                    {
                        crossed=endpointHit=true; firstHit=along; hitX=px; hitY=py;
                        pointNormalX=(qx+along*travelX)/reach; pointNormalY=(qy+along*travelY)/reach;
                    }
                }
            }
            if (distance>=reach && !crossed) return;
            if (crossed || distance<1.0e-7f)
            {
                const float length=std::sqrt(length2);
                dx=length>1.0e-7f && !endpointHit ? -ey/length : pointNormalX;
                dy=length>1.0e-7f && !endpointHit ? ex/length : pointNormalY;
                if ((fromX-hitX)*dx+(fromY-hitY)*dy<0) { dx=-dx; dy=-dy; }
                body.x=hitX+dx*reach; body.y=hitY+dy*reach;
            }
            else
            { dx/=distance; dy/=distance; body.x=cx+dx*reach; body.y=cy+dy*reach; }
            const float surfaceX=std::clamp(motionX/dt,-32.0f,32.0f),surfaceY=std::clamp(motionY/dt,-32.0f,32.0f);
            const float relative=(body.vx-surfaceX)*dx+(body.vy-surfaceY)*dy;
            if (relative<0)
            {
                body.vx-=(1+p.restitution)*relative*dx; body.vy-=(1+p.restitution)*relative*dy;
                const float tangent=((body.vx-surfaceX)*-dy+(body.vy-surfaceY)*dx)*p.friction;
                body.vx+=tangent*dy; body.vy-=tangent*dx;
                body.vx=std::clamp(body.vx,-32.0f,32.0f); body.vy=std::clamp(body.vy,-32.0f,32.0f);
            }
        };
        // R1: a changed-axis box against a mesh may use the rotating class only
        // if no positional link or body-pair operation can move it before its
        // pass-zero certificate. Otherwise, or when no exact width was admitted,
        // refuse the whole step here, before any of those operations runs.
        std::array<bool,kParticleSolidMaxBodies> changedAxis {};
        bool rotatingStep=false;
        if (solid && p.bodyShape==2 && (geometryFrame.meshIdentity || previousGeometry.meshIdentity))
            for (int i=0;i<count;++i) {
                const auto at=static_cast<std::size_t>(i);
                changedAxis[at]=solidBox(previousBodies[at],boxHalf).axes!=solidBox(bodies[at],boxHalf).axes;
                rotatingStep=rotatingStep || changedAxis[at];
            }
        if (rotatingStep && (!rotatingLimbs || (p.linkConstraint==1 && linkCount>0) || (p.collisionMode==2 && count>1))) {
            if (diagnostic) *diagnostic="Rotating box triangle sweeps are unsupported; the replay interval is refused";
            return false;
        }
        for (int pass = 0; pass < 4; ++pass)
        {
            const auto passBodies=bodies;
            if (p.linkConstraint == 1)
                for (int i = 0; i < linkCount; ++i)
                {
                    const auto& link = links[static_cast<std::size_t>(i)];
                    auto& a = bodies[static_cast<std::size_t>(link.a)];
                    auto& b = bodies[static_cast<std::size_t>(link.b)];
                    const float dx = b.x - a.x, dy = b.y - a.y, dz=b.z-a.z;
                    const float distance = solid ? std::hypot(std::hypot(dx,dy),dz):std::hypot(dx,dy);
                    if (distance < 1.0e-6f) continue;
                    const float correction = (distance - link.rest) * 0.5f / distance;
                    a.x += dx * correction; a.y += dy * correction;
                    b.x -= dx * correction; b.y -= dy * correction;
                    a.z+=dz*correction; b.z-=dz*correction;
                    const float relative = ((b.vx - a.vx) * dx + (b.vy - a.vy) * dy+(b.vz-a.vz)*dz) * 0.5f / (distance * distance);
                    a.vx += dx * relative; a.vy += dy * relative;
                    b.vx -= dx * relative; b.vy -= dy * relative;
                    a.vz+=dz*relative; b.vz-=dz*relative;
                }
            if (p.collisionMode == 2)
            {
                for (int a = 0; a < count; ++a)
                    for (int b = a + 1; b < count; ++b)
                        if (solid) {
                            if (p.bodyShape==2) {
                                if (pass!=0 || !solidBoxSweep(bodies[a],bodies[b],
                                    previousBodies[a],previousBodies[b],boxHalf,p))
                                    solidBoxContact(bodies[a],bodies[b],boxHalf,p);
                            }
                            else {
                                if (pass==0) solidSphereSweep(bodies[a],&bodies[b],solidPosition(previousBodies[a]),
                                    solidPosition(previousBodies[b]),solidPosition(bodies[b]),p.bodyRadius,{},p);
                                solidSphereContact(bodies[a],&bodies[b],solidPosition(bodies[b]),p.bodyRadius,{},p);
                            }
                        } else contact(bodies[static_cast<std::size_t>(a)], bodies[static_cast<std::size_t>(b)]);
            }
            for (int i = 0; i < count; ++i)
            {
                if (solid) {
                    // Later constraint passes also sweep their correction path
                    // against the final static surface, rather than tunnelling.
                    const auto& contactPrevious=pass==0 ? previousBodies[i]:passBodies[i];
                    SolidTriangleContactImpulse impulse;
                    // Issue the once-only view for a changed-axis box on pass
                    // zero and retire it as soon as the call returns.
                    const auto at=static_cast<std::size_t>(i);
                    const auto* view=pass==0 && changedAxis[at] ? rotations.issue(at,previousBodies[at],passBodies[at]):nullptr;
                    const SolidRotatingSweep rotating{view,&rotatingWorkspace,rotatingLimbs};
                    const bool contacted=solidGeometryContact(bodies[i],contactPrevious,geometryFrame,previousGeometry,dt,p,
                        pass==0,p.bodyShape==2 ? &boxHalf : nullptr,&triangleWork,diagnostic,true,
                        waterContact ? &impulse:nullptr,waterMeshIdentity,view ? &rotating:nullptr);
                    if (view) rotations.retire();
                    if (!contacted) return false;
                    if (impulse.valid && waterContact && !waterContact(impulse,geometryFrame,error_)) return false;
                    continue;
                }
                for (int edge=0;edge<geometryFrame.count;++edge)
                    geometryContact(bodies[static_cast<std::size_t>(i)],previousBodies[static_cast<std::size_t>(i)],edge);
                walls(bodies[static_cast<std::size_t>(i)]);
            }
        }
        return true;
    }

    bool issueReceipt()
    {
        try {
            receipt_=std::shared_ptr<const ParticleBodyReplayReceipt>(new ParticleBodyReplayReceipt(
                input,p,clipSeconds,aspect,window,result,publishedSolid,bodySampler,callerRetained_));
            return true;
        } catch (const std::bad_alloc&) {
            error_="Dynamic body replay receipt allocation failed before publication"; return false;
        }
    }

    bool finish(const ParticlePriorWaterGeometryView* priorWater)
    {
        if (p.historicalReplay)
        {
            if (!sampleHistory(startTicks + ageTicks, true, priorWater)) return false;
        }
        for (int i = 0; i < count; ++i)
        {
            const auto& body = bodies[static_cast<std::size_t>(i)];
            result[static_cast<std::size_t>(i)] = { body.x, body.y, 1, body.hue };
        }
        if (wantPointTimes) {
            publishedTimes.birthSeconds=births; publishedTimes.lifetimeSeconds=lifetimes;
            publishedTimes.sampleSeconds=p.historicalReplay ? double(startTicks+ageTicks)/ticksPerSecond : p.resetTime+age;
        }
        if (wantSolidState || dynamicBody) {
            publishedSolid.bodies=bodies; publishedSolid.count=count; publishedSolid.colliders=geometryFrame;
            publishedSolid.terminalBodySample=bodySample;
        }
        if (dynamicBody && !issueReceipt()) return false;
        completed_=true; return true;
    }
};

// The inline rotating-plane storage makes the replay state too large for the
// default stacks of worker threads that render particles, so callers own it on the heap.
inline std::unique_ptr<ParticleBodyReplayState> makeParticleBodyReplayState(std::string& error)
{
    try { return std::make_unique<ParticleBodyReplayState>(); }
    catch (const std::bad_alloc&) { error="Particle body replay state allocation failed"; return nullptr; }
}

// The water height solver retains local XY and moves vertices along local Z.
// Contact momentum in that coordinate is conserved using each vertex's lumped
// mass: authored mass per simulation-space area times its triangle area share.
inline bool replayWaterCoupledBodies(const ParticleParams& input, double clipSeconds, float aspect,
    const canonicalblockc::CanonicalBlockCFrame* frame, ParticleBodyState& output,
    ParticleSolidState& solids, ParticlePointReplayTimes* pointTimes, std::string& error,
    std::vector<double>* waterVelocity = nullptr, WaterCoupledFrame* waterFrame = nullptr)
{
    using namespace videowire::geometry;
    const auto refuse=[&](const char* reason) { error=reason; return false; };
    if (!input.geometryBinding || !input.geometryBinding->mesh || !input.geometryBinding->meshFamily.empty() || !input.history
        || input.motionMode!=2 || !input.historicalReplay || input.simulationSpace!=1
        || input.bodyShape!=0 || input.resetMode!=1 || input.geometryBinding->points || input.count<1 || input.count>63
        || !std::isfinite(input.historyProjectSeconds) || input.historyProjectSeconds!=clipSeconds
        || !particleHistoryReady(input,error,clipSeconds))
        return refuse("Water body coupling requires one exact water mesh and project-zero Solid 3D sphere history");
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> borrowedFrame(
        std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>{},frame);
    if (!canonicalblockc::valid(borrowedFrame))
        return refuse("Water body coupling requires the current canonical project frame");
    auto expected=input.history->key;
    expected.fps=frame->key.fps;
    const double frameAtTime=input.historyProjectSeconds*expected.fps;
    if (!std::isfinite(frameAtTime) || frameAtTime<0
        || frameAtTime>=double(std::numeric_limits<std::int64_t>::max()))
        return refuse("Water body coupling project time exceeds its canonical frame clock");
    expected.frame=std::llround(frameAtTime);
    expected.beat=double(float(input.history->timeline.secondsToBeat(double(expected.frame)/expected.fps)));
    if (!(expected==frame->key))
        return refuse("Water body coupling history does not match the current canonical frame");
    const auto& descriptor=*input.geometryBinding->mesh;
    const auto& operations=descriptor.operations;
    const auto water=std::find_if(operations.begin(),operations.end(),[](const auto& op)
        { return op.code==OperationCode::fieldWaterHeight; });
    if (water==operations.end() || water->field.values[4]<=0
        || std::count_if(operations.begin(),operations.end(),[](const auto& op)
            { return op.code==OperationCode::fieldWaterHeight; })!=1
        || operations.end()-water!=3 || operations.back().code!=OperationCode::vectorDisplacement
        || operations.back().outputStableId!=descriptor.stableId || operations.back().parameters[0]!=1
        || !descriptor.spectrumFields.empty() || !descriptor.scoreFields.empty())
        return refuse("Water body coupling requires the exact terminal Water Surface Simulation mesh");
    for (auto at=operations.begin();at!=water;++at)
        if (at->code==OperationCode::fieldTimeline || at->code==OperationCode::fieldScoreSample || at->code==OperationCode::fieldAudioSpectrum || (at->retainedMesh && !at->retainedMesh->importedAnimation.empty()))
            return refuse("Water body coupling requires a static retained rest grid");
    GeometryData rest;
    WaterSurfaceGrid grid;
    std::size_t requests=0;
    RuntimeFieldEvaluation evaluation; evaluation.timelineSeconds=0;
    evaluation.waterHeightAt=[&](const auto& op,const auto& mesh,const auto& requestedGrid,
        double seconds,auto& heights,std::string& diagnostic) {
        if (++requests!=1 || op.stableId!=water->stableId || seconds!=0) {
            diagnostic="Water body rest request does not match the bound operation"; return false;
        }
        rest=mesh; grid=requestedGrid; heights.assign(mesh.positions.size(),0); return true;
    };
    MaterializedInstancePlan restReplay;
    const auto* terminal=replayGeometryMesh(descriptor,restReplay,error,&evaluation);
    if (!terminal || requests!=1 || !equal(*terminal,rest)
        || grid.columns!=grid.rows || (grid.columns!=8 && grid.columns!=9)
        || rest.positions.size()>256 || rest.indices.size()>384)
        return refuse("Water body coupling requires one unchanged 8 by 8 or 9 by 9 rest grid within the 128 triangle bound");
    WaterSurfaceReplayState wave;
    if (!wave.prepare(*input.history,*water,rest,grid,input.historyProjectSeconds,error)) return false;
    const auto scale=double(input.geometryBinding->scale);
    if (!std::isfinite(scale) || scale<=0 || input.geometryBinding->projection!=0)
        return refuse("Water body coupling requires positive uniform XYZ simulation scale");
    std::vector<double> vertexMass(rest.positions.size(),0);
    for (std::size_t triangle=0;triangle<rest.indices.size();triangle+=3) {
        const auto a=rest.positions[rest.indices[triangle]],b=rest.positions[rest.indices[triangle+1]],c=rest.positions[rest.indices[triangle+2]];
        const double area=0.5*std::abs((double(b.x)-a.x)*(double(c.y)-a.y)-(double(b.y)-a.y)*(double(c.x)-a.x))*scale*scale;
        const double mass=water->field.values[4]*area/3;
        if (!std::isfinite(mass) || mass<=0) return refuse("Water contact vertex mass is not finite and positive");
        for (int corner=0;corner<3;++corner) vertexMass[rest.indices[triangle+corner]]+=mass;
    }
    auto parameters=input;
    parameters.preparedBodies.reset(); parameters.preparedSolids.reset();
    double resetTime=parameters.resetTime;
    if (parameters.history->parameterAt && parameters.historyNodeId>0
        && !parameters.history->parameterAt("clip"+std::to_string(parameters.historyClipId)+"/visual"
            +std::to_string(parameters.historyNodeId)+"/resetTime",parameters.historyProjectSeconds,resetTime,error)) return false;
    if (!std::isfinite(resetTime) || resetTime!=0)
        return refuse("Water body coupling requires a finite zero reset time");
    parameters.resetTime=0;
    const auto window=particleReplayWindow(parameters,clipSeconds);
    if (window.start!=0 || window.age!=wave.requestedTicks())
        return refuse("Water body coupling requires an uncycled project-zero particle replay window");
    std::int64_t waterStart=0;
    std::uint64_t viewIdentity=1;
    auto previousHeight=wave.height();
    const auto view=[&]() { return ParticlePriorWaterGeometryView{*water,rest,grid,previousHeight,waterStart,
        waterStart+kParticleReplayStepTicks,viewIdentity,&wave.height()}; };
    auto initial=view();
    auto ownedBodies=makeParticleBodyReplayState(error);
    if (!ownedBodies) return false;
    auto& bodies=*ownedBodies;
    if (!bodies.prepare(parameters,clipSeconds,aspect,frame,error,true,pointTimes!=nullptr,&initial)) return false;
    std::vector<double> impulses(rest.positions.size(),0);
    std::array<std::pair<std::int64_t,std::uint64_t>,63> contactBodies {};
    std::size_t contacts=0;
    bodies.waterMeshIdentity=descriptor.stableId;
    bodies.waterContact=[&](const auto& contact,const auto& mesh,std::string& diagnostic) {
        if (contacts>=contactBodies.size()
            || contact.meshIdentity!=descriptor.stableId || contact.triangle>=rest.indices.size()/3
            || mesh.meshIdentity!=descriptor.stableId || mesh.meshVertexCount!=rest.positions.size()) {
            diagnostic="Water contact exceeds its bounded exact mesh request"; return false;
        }
        const auto identity=std::make_pair(contact.bodyIdentity,contact.bodyGeometryIdentity);
        if (std::find(contactBodies.begin(),contactBodies.begin()+contacts,identity)!=contactBodies.begin()+contacts) {
            diagnostic="Water coupling supports one momentum contact per body in each fixed interval"; return false;
        }
        contactBodies[contacts++]=identity;
        double sum=0;
        for (int corner=0;corner<3;++corner) {
            const auto index=rest.indices[3*contact.triangle+corner];
            if (contact.vertexIdentities[corner]!=rest.vertexIds[index]
                || mesh.meshTriangles[contact.triangle][corner]!=index
                || mesh.meshVertexIdentities[index]!=rest.vertexIds[index]
                || !std::isfinite(contact.weights[corner]) || contact.weights[corner]<0 || contact.weights[corner]>1) {
                diagnostic="Water contact lost its exact ordered vertex identities or barycentrics"; return false;
            }
            sum+=contact.weights[corner];
        }
        if (std::abs(sum-1)>1e-8 || !std::isfinite(contact.time) || contact.time<0 || contact.time>1
            || !std::all_of(contact.impulse.begin(),contact.impulse.end(),[](double value) { return std::isfinite(value); })) {
            diagnostic="Water contact momentum is not finite and normalized"; return false;
        }
        for (int corner=0;corner<3;++corner) {
            const auto index=rest.indices[3*contact.triangle+corner];
            impulses[index]-=contact.impulse[2]*contact.weights[corner]/(vertexMass[index]*scale);
            if (!std::isfinite(impulses[index])) { diagnostic="Water contact velocity overflowed"; return false; }
        }
        return true;
    };
    bool intervalPrepared=false;
    while (!bodies.complete()) {
        std::int64_t start=0,end=0;
        if (!bodies.nextIntervalTicks(start,end) || start<waterStart || end>waterStart+kParticleReplayStepTicks)
            return refuse("Water and body replay intervals do not share the exact rational clock");
        if (!intervalPrepared) {
            previousHeight=wave.height();
            if (!wave.complete() && !wave.advance(error)) return false;
            ++viewIdentity; intervalPrepared=true;
        }
        auto prior=view();
        if (!bodies.advanceOne(error,&prior)) return false;
        if (end==waterStart+kParticleReplayStepTicks || bodies.complete()) {
            // Observed momentum changes velocity now, then displacement during
            // the following water step. It cannot alter the contact endpoints.
            if (!wave.applyBodyVelocity(impulses,error)) return false;
            std::fill(impulses.begin(),impulses.end(),0);
            if (end==waterStart+kParticleReplayStepTicks) {
                waterStart=end; intervalPrepared=false; contacts=0; ++viewIdentity;
            }
        }
    }
    if (!wave.complete()) return refuse("Water and body replay did not complete together");
    ParticleBodyState staged;
    ParticleSolidState stagedSolids;
    ParticlePointReplayTimes stagedTimes;
    if (!bodies.publish(staged,error,&stagedSolids,pointTimes ? &stagedTimes:nullptr)) return false;
    // Re-sample the final collider from the same completed wave state, rather
    // than publishing the frozen prior-interval collider used by the body step.
    previousHeight=wave.height();
    auto finalView=view();
    if (!sampleParticleGeometry(parameters,wave.requestedTicks(),stagedSolids.colliders,error,&finalView)) return false;
    if (waterVelocity) *waterVelocity=wave.velocity();
    if (waterFrame) *waterFrame=WaterCoupledFrame{*water,rest,grid,input.historyProjectSeconds,wave.height()};
    output=std::move(staged); solids=std::move(stagedSolids);
    if (pointTimes) *pointTimes=stagedTimes;
    error.clear(); return true;
}

inline bool hasBodyCoupledWater(const ParticleParams& input)
{
    return input.geometryBinding && input.geometryBinding->mesh
        && std::any_of(input.geometryBinding->mesh->operations.begin(),input.geometryBinding->mesh->operations.end(),
            [](const auto& operation) { return operation.code==videowire::geometry::OperationCode::fieldWaterHeight
                && operation.field.values[4]>0; });
}

inline ParticleBodyState replayParticleBodies(
    const ParticleParams& input, double clipSeconds, float aspect,
    const canonicalblockc::CanonicalBlockCFrame* frame, std::string* diagnostic = nullptr,
    ParticleSolidState* solidState = nullptr, ParticlePointReplayTimes* pointTimes = nullptr,
    std::shared_ptr<const ParticleBodyReplayReceipt>* receipt = nullptr,
    const std::shared_ptr<const ParticleBodyCallerLease>& callerGeometryResources = {})
{
    if (diagnostic) diagnostic->clear();
    if (solidState) *solidState={};
    if (pointTimes) *pointTimes={};
    ParticleBodyState result {};
    for (auto& value : result) value={-10,-10,0,0};
    if (!requiresAuthenticatedParticleBodyReplay(input) && !pointTimes && input.preparedBodies && (!solidState || input.preparedSolids)) {
        if (solidState) *solidState=*input.preparedSolids;
        return *input.preparedBodies;
    }
    if (hasBodyCoupledWater(input) && requiresAuthenticatedParticleBodyReplay(input)) {
        if (diagnostic) *diagnostic="Dynamic body appearance has no coupled Water replay schedule";
        return result;
    }
    if (hasBodyCoupledWater(input)) {
        ParticleSolidState solids;
        std::string error;
        if (!replayWaterCoupledBodies(input,clipSeconds,aspect,frame,result,solids,pointTimes,error)) {
            if (error.empty()) error="Coupled Water body replay failed without a diagnostic";
            if (diagnostic) *diagnostic=error;
            return result;
        }
        if (solidState) *solidState=std::move(solids);
        return result;
    }
    std::string error;
    auto ownedState=makeParticleBodyReplayState(error);
    if (!ownedState) { if (diagnostic) *diagnostic=error; return result; }
    auto& state=*ownedState;
    if (!state.prepare(input,clipSeconds,aspect,frame,error,solidState!=nullptr,pointTimes!=nullptr,nullptr,callerGeometryResources)) {
        if (diagnostic) *diagnostic=error;
        return result;
    }
    while (!state.complete())
        if (!state.advanceOne(error)) {
            if (error.empty()) error="Particle body replay step failed without a diagnostic";
            if (diagnostic) *diagnostic=error;
            return result;
        }
    if (!state.publish(result,error,solidState,pointTimes)) {
        // Callers treat an empty diagnostic as success and draw the fill value.
        if (error.empty()) error="Particle body replay publication failed without a diagnostic";
        if (diagnostic) *diagnostic=error;
    }
    else if (receipt) *receipt=state.receipt();
    return result;
}

// Planar shaders read a fixed 64-entry body array. Planar replay never fills
// more than kParticlePlanarMaxBodies entries.
inline std::array<std::array<float, 4>, kParticlePlanarMaxBodies> planarParticleBodies(
    const ParticleBodyState& bodies) noexcept
{
    std::array<std::array<float, 4>, kParticlePlanarMaxBodies> planar {};
    std::copy_n(bodies.begin(), planar.size(), planar.begin());
    return planar;
}

inline bool replayParticlePoints(const ParticleParams& input, double clipSeconds, float aspect,
    const canonicalblockc::CanonicalBlockCFrame* frame,
    videowire::geometry::StableId outputId, const videowire::geometry::DeferredPointsSource& binding,
    videowire::geometry::ValueDescriptor& result, std::string& error)
{
    using namespace videowire::geometry;
    error.clear();
    if (!validStableId(outputId) || !validStableId(binding.sourceStableId) || binding.sourceRevision==0
        || input.motionMode!=2 || !input.historicalReplay || input.count<1
        || input.count>particleMaxBodies(input.simulationSpace)
        || binding.capacity!=static_cast<std::size_t>(input.count) || !std::isfinite(clipSeconds)
        || !std::isfinite(aspect) || aspect<=0 || !particleHistoryReady(input,error,clipSeconds)) {
        if (error.empty()) error="Points3D requires Coupled bodies with owned project history and at most 64 planar or 128 Solid 3D bodies";
        return false;
    }
    ParticleSolidState state;
    ParticlePointReplayTimes times;
    replayParticleBodies(input,clipSeconds,aspect,frame,&error,&state,&times);
    if (!error.empty()) return false;
    ValueDescriptor value;
    value.carrier=CarrierKind::points3D; value.stableId=outputId;
    value.sourceStableId=binding.sourceStableId; value.sourceRevision=binding.sourceRevision;
    value.data=PointsData{};
    value.attributes={{{1,"velocity",ValueType::vector,Domain::point,Interpolation::nearest},{}},
                      {{2,"age",ValueType::floatValue,Domain::point,Interpolation::nearest},{}},
                      {{3,"lifetime",ValueType::floatValue,Domain::point,Interpolation::nearest},{}}};
    auto currentParameters=input;
    if (!sampleParticleParameters(input,currentParameters,times.sampleSeconds,error)) return false;
    auto& points=std::get<PointsData>(value.data).points;
    std::set<StableId> identities;
    for (int i=0;i<state.count;++i) {
        const auto& body=state.bodies[static_cast<std::size_t>(i)];
        if (body.geometryIdentity==0 && body.noteRow<0) {
            error="Particle Points3D emitters require explicit stable point identities"; return false;
        }
        // Signed projected note IDs use a reversible, nonzero zigzag mapping.
        const auto id=body.geometryIdentity!=0 ? body.geometryIdentity : body.identity>0
            ? static_cast<StableId>(body.identity)*2
            : static_cast<StableId>(-(body.identity+1))*2+1;
        if (!validStableId(id) || !identities.insert(id).second) {
            error="Particle identities must be unique and representable as Points3D"; return false;
        }
        points.push_back({{body.x,body.y,body.z},id,currentParameters.bodyRadius,
            {body.orientation[0],body.orientation[1],body.orientation[2],body.orientation[3]}});
        value.attributes[0].elements.push_back({{body.vx,body.vy,body.vz,0}});
        value.attributes[1].elements.push_back({{std::max(0.0,times.sampleSeconds-times.birthSeconds[i]),0,0,0}});
        value.attributes[2].elements.push_back({{times.lifetimeSeconds[i],0,0,0}});
    }
    result=std::move(value); return true;
}
}
