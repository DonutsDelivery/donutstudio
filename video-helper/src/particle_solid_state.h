#pragma once

#include "particle_geometry.h"
#include "particle_body_caller_admission.h"
#include "particle_body_geometry_sample.h"
#include "geometry_core_admission.h"
#include "particle_rotating_plane_response.h"
#include "../../shared/ParticleSolidBoxExtents.h"
#include <limits>
#include <cfenv>

namespace videorender
{
struct ParticleRigidBody
{
    float x=0,y=0,vx=0,vy=0,tx=0,ty=0,hue=0;
    int noteRow=-1;
    std::int64_t identity=0;
    std::uint64_t geometryIdentity=0;
    float z=0,vz=0,tz=0;
    std::array<float,4> orientation {0,0,0,1};
    std::array<float,3> angular {};
};

struct ParticleSolidState
{
    // An escaped publication keeps its caller output storage reserved.
    std::shared_ptr<const ParticleBodyCallerLease> callerLease;
    std::array<ParticleRigidBody,kParticleSolidMaxBodies> bodies {};
    int count=0;
    ParticleGeometryFrame colliders;
    std::shared_ptr<const ParticleBodyGeometrySample> terminalBodySample;
    ParticleSolidState()=default;
    ParticleSolidState(const ParticleSolidState&)=default;
    ParticleSolidState(ParticleSolidState&&)=default;
    ParticleSolidState& operator=(const ParticleSolidState& other)
    {
        if (this!=&other) {
            // Hold the previous reservation until every previous payload field
            // has been replaced, including its escaped terminal sample owner.
            auto retired=std::move(callerLease);
            bodies=other.bodies; count=other.count; colliders=other.colliders;
            terminalBodySample=other.terminalBodySample; callerLease=other.callerLease;
        }
        return *this;
    }
    ParticleSolidState& operator=(ParticleSolidState&& other) noexcept
    {
        if (this!=&other) {
            auto retired=std::move(callerLease);
            bodies=std::move(other.bodies); count=other.count; colliders=std::move(other.colliders);
            terminalBodySample=std::move(other.terminalBodySample); callerLease=std::move(other.callerLease);
        }
        return *this;
    }
};

inline bool requiresAuthenticatedParticleBodyReplay(const ParticleParams& p)
{
    return hasDynamicParticleBody(p) || (p.preparedSolids && p.preparedSolids->terminalBodySample);
}

class ParticleBodyReplayState;
// The factory admits this payload. Configurable inputs identify a requested
// replay, but confer no completed-state authority.
struct ParticleBodyReplayContext final
{
private:
    struct Payload;
    const std::shared_ptr<const ParticleBodyCallerLease> callerLease_;
    explicit ParticleBodyReplayContext(std::shared_ptr<const ParticleBodyCallerLease> lease)
        : callerLease_(std::move(lease)),caller(callerLease_->ledger) {}
    bool admitted() const noexcept;
    friend class ParticleBodyReplayState;
    friend std::shared_ptr<ParticleBodyReplayContext> makeParticleBodyReplayContext(
        const std::shared_ptr<ParticleBodyCallerLedger>&,std::string&);
public:
    ParticleBodyReplayContext(const ParticleBodyReplayContext&)=delete;
    ParticleBodyReplayContext(ParticleBodyReplayContext&&)=delete;
    ParticleBodyReplayContext& operator=(const ParticleBodyReplayContext&)=delete;
    ParticleBodyReplayContext& operator=(ParticleBodyReplayContext&&)=delete;
    const std::shared_ptr<ParticleBodyCallerLedger> caller;
    videohelper::geometry::PlanOwnerIdentity owner;
    int width=0, height=0;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frame;
    ParticleBodyGeometryLimits admissionLimits;
    ParticleBodyGeometrySampleLimits sampleLimits;
};
struct ParticleBodyReplayContext::Payload final
{
    // The actual context and its frame owner retire before the reservation.
    const std::shared_ptr<const ParticleBodyCallerLease> lease;
    ParticleBodyReplayContext value;
    explicit Payload(std::shared_ptr<const ParticleBodyCallerLease> admittedLease)
        : lease(std::move(admittedLease)),value(lease) {}
};

// The live layer, local parameters and scalar wrapper outputs coexist with
// replay/receipt payloads. Their bounded inline metadata is reserved here.
inline constexpr std::size_t particleBodyContextLogicalBytes=sizeof(ParticleBodyCallerPayload<ParticleBodyReplayContext>)
    +3*sizeof(ParticleParams)+sizeof(ParticleBodyState)+sizeof(ParticleSolidState)
    +sizeof(ParticleHistorySource)+sizeof(canonicalblockc::CanonicalBlockCFrame);
inline bool ParticleBodyReplayContext::admitted() const noexcept
{
    return callerLease_ && caller && callerLease_->ledger==caller
        && callerLease_->bytes==particleBodyContextLogicalBytes+sizeof(ParticleBodyCallerLease);
}
inline std::shared_ptr<ParticleBodyReplayContext> makeParticleBodyReplayContext(
    const std::shared_ptr<ParticleBodyCallerLedger>& caller,std::string& error)
{
    if (!caller) { error="Body replay has no stable caller ledger"; return {}; }
    auto lease=caller->reserve(particleBodyContextLogicalBytes,sizeof(ParticleBodyReplayContext)+3*sizeof(ParticleParams));
    if (!lease) { error="Body replay context prospective caller admission is short"; return {}; }
    try {
        static_assert(sizeof(ParticleBodyReplayContext::Payload)
            ==sizeof(ParticleBodyCallerPayload<ParticleBodyReplayContext>));
        auto payload=std::make_shared<ParticleBodyReplayContext::Payload>(std::move(lease));
        return std::shared_ptr<ParticleBodyReplayContext>(payload,&payload->value);
    } catch (const std::bad_alloc&) { error="Body replay context allocation failed"; return {}; }
}

class ParticleBodyReplayReceipt final
{
public:
    ParticleBodyReplayReceipt(const ParticleBodyReplayReceipt&)=delete;
    ParticleBodyReplayReceipt& operator=(const ParticleBodyReplayReceipt&)=delete;
    const std::shared_ptr<const ParticleBodyState>& bodies() const noexcept { return bodies_; }
    const std::shared_ptr<const ParticleSolidState>& solids() const noexcept { return solids_; }
    const std::shared_ptr<const ParticleBodyGeometrySampler>& sampler() const noexcept { return sampler_; }
    std::int64_t terminalTick() const noexcept { return window_.start+window_.age; }
    double clipSeconds() const noexcept { return clipSeconds_; }
    float aspect() const noexcept { return aspect_; }
    const ParticleParams& terminalParameters() const noexcept { return terminal_; }

    bool matches(const ParticleParams& p,double clipSeconds,float aspect,
        const canonicalblockc::CanonicalBlockCFrame* frame) const
    {
        // Reserve both possible parameter-copy comparisons and the complete
        // frozen source comparison before any of their loops or copies run.
        const auto validation=callerLease_->ledger->reserve(validationLogicalBytes(),validationWork());
        if (!validation) return false;
        if (!p.bodyReplayContext || p.bodyReplayContext->caller!=callerLease_->ledger
            || p.bodyReplayContext!=request_.bodyReplayContext || !p.history || !p.geometryBinding
            || !(p.bodyReplayContext->owner==owner_) || p.bodyReplayContext->width!=width_
            || p.bodyReplayContext->height!=height_ || p.bodyReplayContext->frame.get()!=frame
            || frame!=frame_.get() || (frame && !(frame->key==frameKey_))
            || p.history!=request_.history || !(p.history->key==historyKey_) || p.history->score!=scoreSource_
            || p.history->audioFrames!=audioFrames_ || p.history->audioSampleRate!=audioSampleRate_
            || p.history->analysisSources!=analysisSources_ || p.history->parameterFrameSeconds!=parameterFrameSeconds_
            || p.geometryBinding!=request_.geometryBinding || p.geometryRevision!=request_.geometryRevision
            || p.historyClipId!=request_.historyClipId || p.historyNodeId!=request_.historyNodeId
            || p.historyProjectSeconds!=request_.historyProjectSeconds || clipSeconds!=clipSeconds_ || aspect!=aspect_)
            return false;
        const auto& binding=*p.geometryBinding;
        if (binding.body!=binding_.body || binding.bodySource!=binding_.bodySource
            || binding.emitters!=binding_.emitters || binding.attractors!=binding_.attractors
            || binding.mesh!=binding_.mesh || binding.points!=binding_.points || binding.scale!=binding_.scale
            || binding.centerX!=binding_.centerX || binding.centerY!=binding_.centerY
            || binding.centerZ!=binding_.centerZ || binding.projection!=binding_.projection
            || binding.meshFamily!=binding_.meshFamily) return false;
        if (sampler_) {
            const auto& admission=solids_->terminalBodySample->admission();
            std::size_t visits=0;
            if (!admission->recipe().sourceComparison.total(visits)
                || !admission->matchesExactSource(binding.bodySource,visits)) return false;
        }
        return sameParameters(p,request_) || sameParameters(p,terminal_);
    }

    static constexpr std::size_t validationLogicalBytes() noexcept
    { return 2*sizeof(ParticleParams)+2*sizeof(std::array<float*,visualsimulation::parameters.size()>); }
    std::size_t validationWork() const noexcept
    {
        std::size_t visits=0;
        if (sampler_ && (!solids_->terminalBodySample
            || !solids_->terminalBodySample->admission()->recipe().sourceComparison.total(visits)
            || visits>std::numeric_limits<std::size_t>::max()-6*sizeof(ParticleParams)
                -4*sizeof(std::array<float*,visualsimulation::parameters.size()>)))
            return std::numeric_limits<std::size_t>::max();
        // Byte visits conservatively include inline array/scalar comparisons,
        // automation pointer tables and two calls that each copy two Params.
        return visits+6*sizeof(ParticleParams)+4*sizeof(std::array<float*,visualsimulation::parameters.size()>);
    }
    static constexpr std::size_t retainedLogicalBytes() noexcept;

    bool matchesPrepared(const ParticleParams& p,double clipSeconds,float aspect,
        const canonicalblockc::CanonicalBlockCFrame* frame) const
    {
        return matches(p,clipSeconds,aspect,frame) && p.preparedBodyReplay.get()==this
            && p.preparedBodies==bodies_ && p.preparedSolids==solids_;
    }
    bool matchesRender(const ParticleParams& p,const videohelper::geometry::PlanOwnerIdentity& owner,
        int nodeId,int width,int height) const
    {
        return matchesPrepared(p,clipSeconds_,aspect_,p.bodyReplayContext ? p.bodyReplayContext->frame.get():nullptr)
            && owner==owner_ && nodeId==request_.historyNodeId && width==width_ && height==height_;
    }
private:
    ParticleBodyReplayReceipt(const ParticleParams& request,const ParticleParams& terminal,
        double seconds,float aspect,ParticleReplayWindow window,const ParticleBodyState& bodies,
        const ParticleSolidState& solids,std::shared_ptr<const ParticleBodyGeometrySampler> sampler,
        std::shared_ptr<const ParticleBodyCallerLease> lease)
        : callerLease_(std::move(lease)),request_(request),terminal_(terminal),binding_(*request.geometryBinding),
          owner_(request.bodyReplayContext->owner),width_(request.bodyReplayContext->width),
          height_(request.bodyReplayContext->height),historyKey_(request.history->key),
          frameKey_(request.bodyReplayContext->frame ? request.bodyReplayContext->frame->key:canonicalblockc::FrameKey{}),
          frame_(request.bodyReplayContext->frame),scoreSource_(request.history->score),audioFrames_(request.history->audioFrames),
          audioSampleRate_(request.history->audioSampleRate),analysisSources_(request.history->analysisSources),
          parameterFrameSeconds_(request.history->parameterFrameSeconds),clipSeconds_(seconds),aspect_(aspect),window_(window),
          bodies_(retainParticleBodyCallerPayload<ParticleBodyState>(callerLease_,bodies)),
          solids_(retainParticleBodyCallerPayload<ParticleSolidState>(callerLease_,solids)),sampler_(std::move(sampler))
    {
        // History audio remains borrowed. These snapshots are only compared;
        // no history callback is invoked through a retained receipt.
        std::const_pointer_cast<ParticleSolidState>(solids_)->callerLease=callerLease_;
        request_.preparedBodies.reset(); request_.preparedSolids.reset(); request_.preparedBodyReplay.reset();
        terminal_.preparedBodies.reset(); terminal_.preparedSolids.reset(); terminal_.preparedBodyReplay.reset();
    }
    static bool sameParameters(const ParticleParams& a,const ParticleParams& b)
    {
        auto x=a,y=b;
        const auto xf=particleAutomationFields(x),yf=particleAutomationFields(y);
        for (std::size_t i=0;i<xf.size();++i) if (*xf[i]!=*yf[i]) return false;
        return a.count==b.count && a.spawnTrack==b.spawnTrack && a.seed==b.seed && a.size==b.size
            && a.lifetime==b.lifetime && a.red==b.red && a.green==b.green && a.blue==b.blue && a.alpha==b.alpha
            && a.motionMode==b.motionMode && a.collisionMode==b.collisionMode && a.bodyShape==b.bodyShape
            && a.linkConstraint==b.linkConstraint && a.resetMode==b.resetMode && a.simulationSpace==b.simulationSpace
            && a.spawnZ==b.spawnZ && a.orientationDegrees==b.orientationDegrees && a.angularVelocity==b.angularVelocity
            && a.historicalReplay==b.historicalReplay && a.colliderThickness==b.colliderThickness
            && a.geometryCount==b.geometryCount && a.geometryAnchors==b.geometryAnchors
            && a.geometryIdentities==b.geometryIdentities && a.geometryDepths==b.geometryDepths
            && a.rms==b.rms && a.onset==b.onset && a.onsetAge==b.onsetAge;
    }
    const std::shared_ptr<const ParticleBodyCallerLease> callerLease_;
    ParticleParams request_,terminal_;
    const ParticleGeometryBinding binding_;
    const videohelper::geometry::PlanOwnerIdentity owner_;
    const int width_,height_;
    const canonicalblockc::FrameKey historyKey_,frameKey_;
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frame_;
    const std::shared_ptr<const arbitmod::Score> scoreSource_;
    const std::vector<arbitblockb::FeatureFrame>* const audioFrames_;
    const double audioSampleRate_;
    const std::shared_ptr<const videohelper::AnalysisSources> analysisSources_;
    const double parameterFrameSeconds_;
    const double clipSeconds_;
    const float aspect_;
    const ParticleReplayWindow window_;
    const std::shared_ptr<const ParticleBodyState> bodies_;
    const std::shared_ptr<const ParticleSolidState> solids_;
    const std::shared_ptr<const ParticleBodyGeometrySampler> sampler_;
    friend class ParticleBodyReplayState;
};
inline constexpr std::size_t ParticleBodyReplayReceipt::retainedLogicalBytes() noexcept
{
    // Shared source/sampler payloads have their own ledgers. Only newly retained
    // history/frame metadata and the actual copied caller objects are added.
    return sizeof(ParticleBodyReplayReceipt)+sizeof(ParticleBodyCallerPayload<ParticleBodyState>)
        +sizeof(ParticleBodyCallerPayload<ParticleSolidState>)+sizeof(ParticleHistorySource)
        +sizeof(canonicalblockc::CanonicalBlockCFrame);
}

// One accepted triangle response. Time is the fraction of this contact interval.
struct SolidTriangleContactImpulse
{
    bool valid=false;
    std::int64_t bodyIdentity=0;
    std::uint64_t bodyGeometryIdentity=0,meshIdentity=0;
    std::uint32_t triangle=0;
    std::array<std::uint64_t,3> vertexIdentities{};
    std::array<double,3> weights{},point{},impulse{};
    double time=0;
};

using SolidVector=std::array<float,3>;
inline float solidDot(const SolidVector& a,const SolidVector& b)
{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline SolidVector solidCross(const SolidVector& a,const SolidVector& b)
{ return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
inline SolidVector solidPosition(const ParticleRigidBody& b) { return {b.x,b.y,b.z}; }
inline SolidVector solidVelocity(const ParticleRigidBody& b) { return {b.vx,b.vy,b.vz}; }
inline void solidMove(ParticleRigidBody& b,const SolidVector& n,float distance)
{ b.x+=n[0]*distance; b.y+=n[1]*distance; b.z+=n[2]*distance; }
inline void solidVelocityAdd(ParticleRigidBody& b,const SolidVector& delta)
{
    b.vx=std::clamp(b.vx+delta[0],-32.0f,32.0f);
    b.vy=std::clamp(b.vy+delta[1],-32.0f,32.0f);
    b.vz=std::clamp(b.vz+delta[2],-32.0f,32.0f);
}

struct SolidBoxProxy
{
    SolidVector half {};
    std::array<SolidVector,3> axes {};
};

// The renderer centers the static mesh, divides by its enclosing radius, then
// scales by bodyRadius. Use the same bound for the opt-in contact proxy. The
// bound itself is computed in shared/ParticleSolidBoxExtents.h, which the
// plugin lowering gate also uses.
inline bool solidBodyBoxHalfExtents(const ParticleParams& p,SolidVector& half)
{
    if (!p.geometryBinding || !p.geometryBinding->body) return false;
    return videowire::geometry::particleSolidBoxHalfExtents(
        std::get<videowire::geometry::GeometryData>(p.geometryBinding->body->data),p.bodyRadius,half);
}

inline SolidBoxProxy solidBox(const ParticleRigidBody& body,const SolidVector& half)
{
    const auto& q=body.orientation;
    const float x=q[0],y=q[1],z=q[2],w=q[3];
    return {half, {{{1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)},
                    {2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)},
                    {2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)}}}};
}

inline float solidBoxRadius(const SolidBoxProxy& box,const SolidVector& direction)
{
    float radius=0;
    for (int i=0;i<3;++i) radius+=box.half[i]*std::abs(solidDot(box.axes[i],direction));
    return radius;
}

inline SolidVector solidBoxSupport(const ParticleRigidBody& body,const SolidBoxProxy& box,
                                   const SolidVector& direction)
{
    auto point=solidPosition(body);
    for (int i=0;i<3;++i) {
        const float facing=solidDot(box.axes[i],direction);
        const float distance=std::abs(facing)<1.0e-6f ? 0.0f
            : box.half[i]*(facing<0 ? -1.0f:1.0f);
        for (int j=0;j<3;++j) point[j]+=box.axes[i][j]*distance;
    }
    return point;
}

inline SolidVector solidBoxInverseInertia(const SolidBoxProxy& box,float inverseMass,
                                         const SolidVector& vector)
{
    SolidVector result {};
    for (int i=0;i<3;++i) {
        const float inertia=3*inverseMass/(box.half[(i+1)%3]*box.half[(i+1)%3]
            +box.half[(i+2)%3]*box.half[(i+2)%3]);
        const float component=inertia*solidDot(vector,box.axes[i]);
        for (int j=0;j<3;++j) result[j]+=component*box.axes[i][j];
    }
    return result;
}

inline void solidBoxContact(ParticleRigidBody& a,ParticleRigidBody& b,
                            const SolidVector& half,const ParticleParams& p)
{
    const auto boxA=solidBox(a,half),boxB=solidBox(b,half);
    const auto centerA=solidPosition(a),centerB=solidPosition(b);
    SolidVector delta {};
    for (int i=0;i<3;++i) delta[i]=centerA[i]-centerB[i];
    float penetration=std::numeric_limits<float>::max();
    SolidVector normal {};
    const auto probe=[&](SolidVector axis) {
        const float length=std::sqrt(solidDot(axis,axis));
        if (length<1.0e-6f) return true;
        for (auto& value:axis) value/=length;
        const float overlap=solidBoxRadius(boxA,axis)+solidBoxRadius(boxB,axis)-std::abs(solidDot(delta,axis));
        if (overlap<=0) return false;
        if (overlap<penetration) {
            penetration=overlap;
            normal=axis;
            if (solidDot(delta,normal)<0) for (auto& value:normal) value=-value;
        }
        return true;
    };
    for (int i=0;i<3;++i) if (!probe(boxA.axes[i]) || !probe(boxB.axes[i])) return;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        if (!probe(solidCross(boxA.axes[i],boxB.axes[j]))) return;
    solidMove(a,normal,penetration*0.5f);
    solidMove(b,normal,-penetration*0.5f);
    const auto pointA=solidBoxSupport(a,boxA,{-normal[0],-normal[1],-normal[2]});
    const auto pointB=solidBoxSupport(b,boxB,normal);
    SolidVector armA {},armB {};
    for (int i=0;i<3;++i) {
        const float contact=(pointA[i]+pointB[i])*0.5f;
        armA[i]=contact-solidPosition(a)[i]; armB[i]=contact-solidPosition(b)[i];
    }
    const float inverseMass=1/p.bodyMass;
    const auto effectiveMass=[&](const SolidVector& direction) {
        const auto torqueA=solidCross(armA,direction),torqueB=solidCross(armB,direction);
        return 2*inverseMass+solidDot(solidCross(solidBoxInverseInertia(boxA,inverseMass,torqueA),armA),direction)
            +solidDot(solidCross(solidBoxInverseInertia(boxB,inverseMass,torqueB),armB),direction);
    };
    const auto relativeVelocity=[&]() {
        const auto spinA=solidCross(a.angular,armA),spinB=solidCross(b.angular,armB);
        SolidVector result {};
        for (int i=0;i<3;++i) result[i]=solidVelocity(a)[i]+spinA[i]
            -solidVelocity(b)[i]-spinB[i];
        return result;
    };
    const auto apply=[&](const SolidVector& impulse) {
        SolidVector speed {};
        for (int i=0;i<3;++i) speed[i]=impulse[i]*inverseMass;
        solidVelocityAdd(a,speed);
        for (auto& value:speed) value=-value;
        solidVelocityAdd(b,speed);
        const auto spinA=solidBoxInverseInertia(boxA,inverseMass,solidCross(armA,impulse));
        const auto spinB=solidBoxInverseInertia(boxB,inverseMass,solidCross(armB,impulse));
        for (int i=0;i<3;++i) {
            a.angular[i]=std::clamp(a.angular[i]+spinA[i],-32.0f,32.0f);
            b.angular[i]=std::clamp(b.angular[i]-spinB[i],-32.0f,32.0f);
        }
    };
    const auto velocity=relativeVelocity();
    const float closing=solidDot(velocity,normal);
    if (closing>=0) return;
    const float normalImpulse=-(1+p.restitution)*closing/effectiveMass(normal);
    SolidVector impulse {};
    for (int i=0;i<3;++i) impulse[i]=normal[i]*normalImpulse;
    apply(impulse);
    const auto after=relativeVelocity();
    SolidVector tangent {};
    for (int i=0;i<3;++i) tangent[i]=after[i]-solidDot(after,normal)*normal[i];
    const float length=std::sqrt(solidDot(tangent,tangent));
    if (length>1.0e-7f) {
        for (auto& value:tangent) value/=length;
        const float friction=std::min(p.friction*normalImpulse,length/effectiveMass(tangent));
        for (int i=0;i<3;++i) impulse[i]=-tangent[i]*friction;
        apply(impulse);
    }
}

inline bool solidBoxSweep(ParticleRigidBody& a,ParticleRigidBody& b,
                          const ParticleRigidBody& previousA,const ParticleRigidBody& previousB,
                          const SolidVector& half,const ParticleParams& p)
{
    // Sweep the relative translation against the 15 separating axes. The
    // current step's orientations stay fixed during this bounded contact test.
    const auto boxA=solidBox(a,half),boxB=solidBox(b,half);
    const auto fromA=solidPosition(previousA),fromB=solidPosition(previousB);
    const auto toA=solidPosition(a),toB=solidPosition(b);
    SolidVector origin {},travel {};
    for (int i=0;i<3;++i) {
        origin[i]=fromA[i]-fromB[i];
        travel[i]=(toA[i]-fromA[i])-(toB[i]-fromB[i]);
    }
    float enter=0,leave=1;
    bool separated=false;
    const auto probe=[&](SolidVector axis) {
        const float length=std::sqrt(solidDot(axis,axis));
        if (length<1.0e-6f) return true;
        for (auto& value:axis) value/=length;
        const float radius=solidBoxRadius(boxA,axis)+solidBoxRadius(boxB,axis);
        const float offset=solidDot(origin,axis),speed=solidDot(travel,axis);
        separated=separated || std::abs(offset)>=radius;
        if (std::abs(speed)<1.0e-9f) return std::abs(offset)<radius;
        const float first=(-radius-offset)/speed,last=(radius-offset)/speed;
        enter=std::max(enter,std::min(first,last));
        leave=std::min(leave,std::max(first,last));
        return enter<=leave;
    };
    for (int i=0;i<3;++i) if (!probe(boxA.axes[i]) || !probe(boxB.axes[i])) return false;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        if (!probe(solidCross(boxA.axes[i],boxB.axes[j]))) return false;
    if (!separated || enter>=leave || enter>=1 || solidDot(travel,travel)<1.0e-12f)
        return false;
    // Move just inside the first contact so the existing box resolver applies
    // its positional correction, restitution and angular/friction impulses.
    const float hit=std::min(1.0f,enter+1.0e-5f/std::sqrt(solidDot(travel,travel)));
    for (int i=0;i<3;++i) {
        const float positionA=fromA[i]+hit*(toA[i]-fromA[i]);
        const float positionB=fromB[i]+hit*(toB[i]-fromB[i]);
        if (i==0) { a.x=positionA; b.x=positionB; }
        if (i==1) { a.y=positionA; b.y=positionB; }
        if (i==2) { a.z=positionA; b.z=positionB; }
    }
    solidBoxContact(a,b,half,p);
    return true;
}

inline void solidBoxSurfaceImpulse(ParticleRigidBody& body,const SolidBoxProxy& box,
    const SolidVector& arm,const SolidVector& normal,const SolidVector& surface,const ParticleParams& p)
{
    const float inverseMass=1/p.bodyMass;
    const auto effectiveMass=[&](const SolidVector& direction) {
        const auto torque=solidCross(arm,direction);
        return inverseMass+solidDot(solidCross(solidBoxInverseInertia(box,inverseMass,torque),arm),direction);
    };
    const auto relativeVelocity=[&]() {
        const auto spin=solidCross(body.angular,arm);
        SolidVector velocity {};
        for (int j=0;j<3;++j) velocity[j]=solidVelocity(body)[j]+spin[j]-surface[j];
        return velocity;
    };
    const auto apply=[&](const SolidVector& impulse) {
        SolidVector speed {};
        for (int j=0;j<3;++j) speed[j]=impulse[j]*inverseMass;
        solidVelocityAdd(body,speed);
        const auto spin=solidBoxInverseInertia(box,inverseMass,solidCross(arm,impulse));
        for (int j=0;j<3;++j)
            body.angular[j]=std::clamp(body.angular[j]+spin[j],-32.0f,32.0f);
    };
    const auto velocity=relativeVelocity();
    const float closing=solidDot(velocity,normal);
    if (closing>=0) return;
    const float normalImpulse=-(1+p.restitution)*closing/effectiveMass(normal);
    SolidVector impulse {};
    for (int j=0;j<3;++j) impulse[j]=normal[j]*normalImpulse;
    apply(impulse);
    const auto after=relativeVelocity();
    SolidVector tangent {};
    for (int j=0;j<3;++j) tangent[j]=after[j]-solidDot(after,normal)*normal[j];
    const float length=std::sqrt(solidDot(tangent,tangent));
    if (length>1.0e-7f) {
        for (auto& value:tangent) value/=length;
        const float friction=std::min(p.friction*normalImpulse,length/effectiveMass(tangent));
        for (int j=0;j<3;++j) impulse[j]=-tangent[j]*friction;
        apply(impulse);
    }
}

inline void solidBoxSphereContact(ParticleRigidBody& body,const SolidVector& half,
                                  const SolidVector& center,float radius,
                                  const SolidVector& surface,const ParticleParams& p)
{
    const auto box=solidBox(body,half);
    const auto position=solidPosition(body);
    SolidVector offset {};
    for (int j=0;j<3;++j) offset[j]=center[j]-position[j];
    SolidVector local {},closest {},arm {},normal {};
    for (int i=0;i<3;++i) {
        local[i]=solidDot(offset,box.axes[i]);
        closest[i]=std::clamp(local[i],-half[i],half[i]);
        for (int j=0;j<3;++j) arm[j]+=box.axes[i][j]*closest[i];
    }
    SolidVector separation {};
    for (int j=0;j<3;++j) separation[j]=position[j]+arm[j]-center[j];
    const float distance=std::sqrt(solidDot(separation,separation));
    if (distance>=radius && distance>1.0e-7f) return;
    float penetration=radius-distance;
    if (distance>1.0e-7f) {
        for (int j=0;j<3;++j) normal[j]=separation[j]/distance;
    } else {
        int face=0;
        for (int i=1;i<3;++i)
            if (half[i]-std::abs(local[i])<half[face]-std::abs(local[face])) face=i;
        const float sign=local[face]<0 ? -1.0f:1.0f;
        closest[face]=sign*half[face];
        arm={};
        for (int i=0;i<3;++i) for (int j=0;j<3;++j)
            arm[j]+=box.axes[i][j]*closest[i];
        for (int j=0;j<3;++j) normal[j]=-sign*box.axes[face][j];
        penetration=radius+half[face]-std::abs(local[face]);
    }
    solidMove(body,normal,penetration);
    solidBoxSurfaceImpulse(body,box,arm,normal,surface,p);
}

// solidRotate writes its pre-normalization Euler quaternion to raw, before
// the existing sqrt/divide. It returns false for a zero spin, which leaves the
// pose unchanged and so never needs a changing-axis capture.
inline bool solidRotate(ParticleRigidBody& b,float dt,float drag,std::array<float,4>* raw=nullptr)
{
    auto& q=b.orientation;
    const auto old=q;
    for (auto& w:b.angular) w=std::clamp(w/(1+drag*dt),-32.0f,32.0f);
    const auto& w=b.angular;
    if (w[0]==0 && w[1]==0 && w[2]==0) return false;
    q[0]+=0.5f*dt*(w[0]*old[3]+w[1]*old[2]-w[2]*old[1]);
    q[1]+=0.5f*dt*(-w[0]*old[2]+w[1]*old[3]+w[2]*old[0]);
    q[2]+=0.5f*dt*(w[0]*old[1]-w[1]*old[0]+w[2]*old[3]);
    q[3]-=0.5f*dt*(w[0]*old[0]+w[1]*old[1]+w[2]*old[2]);
    if (raw) *raw=q;
    const auto length=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    for (auto& value:q) value/=length;
    return true;
}

class SolidRotationLedger;
// One body's executed solidRotate step for one replay interval: the raw Euler
// quaternion 16 bytes, scheduled angular 12, dt 4, body and body-geometry
// identities 16, and interval start/end ticks 16. Only the owning ledger can
// construct, fill or read it. The previous pose and scheduled endpoint are not
// copied here; the issued view borrows the replay's own snapshots instead.
class SolidRotationCapture
{
public:
    SolidRotationCapture(const SolidRotationCapture&)=delete;
    SolidRotationCapture& operator=(const SolidRotationCapture&)=delete;
private:
    friend class SolidRotationLedger;
    friend class SolidRotationView;
    SolidRotationCapture()=default;
    std::array<float,4> raw {};
    SolidVector angular {};
    float dt=0;
    std::int64_t identity=0;
    std::uint64_t geometryIdentity=0;
    std::int64_t startTick=0,endTick=0;
};
static_assert(sizeof(SolidRotationCapture)==64 && alignof(SolidRotationCapture)==8,
    "the rotating-plane contract prices one 64-byte capture record per body");

// A pass-zero borrow of one issued capture. It points at the owning ledger
// and at the replay's previous and first scheduled body snapshots, so the
// consumer compares every body input bitwise. Only the ledger mints one.
class SolidRotationView
{
public:
    SolidRotationView(const SolidRotationView&)=delete;
    SolidRotationView& operator=(const SolidRotationView&)=delete;
    // Succeeds at most once per issued capture. On success raw and angular
    // hold the executed solidRotate step for exactly this body and interval.
    bool consume(const ParticleRigidBody& candidate,const ParticleRigidBody& previous,float dt,
                 std::array<float,4>& raw,SolidVector& angular) const;
private:
    friend class SolidRotationLedger;
    SolidRotationView()=default;
    SolidRotationLedger* ledger=nullptr;
    const ParticleRigidBody* previous=nullptr;
    const ParticleRigidBody* scheduled=nullptr;
    std::uint32_t slot=0,generation=0;
};
static_assert(sizeof(void*)!=8 || sizeof(SolidRotationView)==32,
    "the rotating-plane contract prices one 32-byte scoped view");

// Test-only driver for the ledger below. Production code never defines it.
// Only tests/visual_plan_executor_tests.cpp does, to issue a view exactly as
// the replay does for one body and check the caller's exact outcomes.
struct SolidRotationTestIssuer;

// Request-local issuer for rotating coordinate-plane sweeps. In production
// only ParticleBodyReplayState constructs one, inline, so the sizeof-based caller
// scratch and prepare charges include it. Layout: 128 captures 8,192 bytes,
// per-body guard states 1,024, interval descriptor 16, generation 8 and the
// one scoped view 32. Guards move unissued -> recorded -> issued -> consumed.
class SolidRotationLedger
{
    friend class ParticleBodyReplayState;
    friend class SolidRotationView;
    friend struct SolidRotationTestIssuer;
    static constexpr std::uint32_t unissued=0,recorded=1,issued=2,consumed=3;
    struct Guard { std::uint32_t state=unissued,reserved=0; };
    SolidRotationLedger()=default;
    SolidRotationLedger(const SolidRotationLedger&)=delete;
    SolidRotationLedger& operator=(const SolidRotationLedger&)=delete;
    SolidRotationCapture record[kParticleSolidMaxBodies];
    Guard guard[kParticleSolidMaxBodies];
    std::int64_t startTick=0,endTick=0;
    std::uint64_t generation=0;
    SolidRotationView view;

    // Clear every issuance before the interval writes any capture.
    void begin(std::int64_t start,std::int64_t end)
    {
        for (auto& state:guard) state=Guard{};
        startTick=start; endTick=end; ++generation;
        retire();
    }
    // The capture is written inside the actual solidRotate call.
    void rotate(ParticleRigidBody& body,std::size_t slot,float dt,float drag)
    {
        auto& entry=record[slot];
        const bool stepped=solidRotate(body,dt,drag,&entry.raw);
        entry.angular=body.angular; entry.dt=dt;
        entry.identity=body.identity; entry.geometryIdentity=body.geometryIdentity;
        entry.startTick=startTick; entry.endTick=endTick;
        guard[slot].state=stepped ? recorded:unissued;
    }
    // Pass zero only. An unrecorded slot still yields a view, which then fails
    // to consume. The view lives until retire().
    const SolidRotationView* issue(std::size_t slot,const ParticleRigidBody& previous,const ParticleRigidBody& scheduled)
    {
        if (slot<static_cast<std::size_t>(kParticleSolidMaxBodies) && guard[slot].state==recorded) guard[slot].state=issued;
        view.ledger=this; view.previous=&previous; view.scheduled=&scheduled;
        view.slot=static_cast<std::uint32_t>(slot);
        view.generation=static_cast<std::uint32_t>(generation);
        return &view;
    }
    void retire()
    {
        view.ledger=nullptr; view.previous=nullptr; view.scheduled=nullptr;
        view.slot=0; view.generation=0;
    }
};
static_assert(kParticleSolidMaxBodies==128 && (sizeof(void*)!=8 || sizeof(SolidRotationLedger)==9272),
    "rotating capture storage is 8,192 + 1,024 + 16 + 8 + 32 bytes");

inline bool SolidRotationView::consume(const ParticleRigidBody& candidate,const ParticleRigidBody& previousBody,
    float dt,std::array<float,4>& raw,SolidVector& angular) const
{
    // A request has at most 1,712 intervals, so 32 generation bits cannot wrap.
    if (!ledger || !previous || !scheduled || slot>=static_cast<std::uint32_t>(kParticleSolidMaxBodies)
        || generation!=static_cast<std::uint32_t>(ledger->generation)) return false;
    auto& state=ledger->guard[slot].state;
    if (state!=SolidRotationLedger::issued) return false;
    state=SolidRotationLedger::consumed;
    const auto& entry=ledger->record[slot];
    using rotatingexact::floatBits;
    const auto same=[](const auto& a,const auto& b) {
        for (std::size_t i=0;i<a.size();++i) if (floatBits(a[i])!=floatBits(b[i])) return false;
        return true;
    };
    const auto pose=[&](const ParticleRigidBody& a,const ParticleRigidBody& b) {
        return a.identity==b.identity && a.geometryIdentity==b.geometryIdentity
            && floatBits(a.x)==floatBits(b.x) && floatBits(a.y)==floatBits(b.y) && floatBits(a.z)==floatBits(b.z)
            && same(a.orientation,b.orientation);
    };
    // Every body input the certificate and response read: identities, both
    // centers and orientations, stored velocity, angular state, dt and ticks.
    if (!pose(previousBody,*previous) || !pose(candidate,*scheduled)
        || floatBits(candidate.vx)!=floatBits(scheduled->vx) || floatBits(candidate.vy)!=floatBits(scheduled->vy)
        || floatBits(candidate.vz)!=floatBits(scheduled->vz) || !same(candidate.angular,scheduled->angular)
        || entry.identity!=candidate.identity || entry.geometryIdentity!=candidate.geometryIdentity
        || floatBits(entry.dt)!=floatBits(dt) || !same(entry.angular,scheduled->angular)
        || entry.startTick!=ledger->startTick || entry.endTick!=ledger->endTick) return false;
    raw=entry.raw; angular=entry.angular;
    return true;
}

// Pass-zero rotating sweep inputs. Only the view carries authority. limbs is
// the request's preflight-admitted exact width, or zero when unavailable.
struct SolidRotatingSweep
{
    const SolidRotationView* view=nullptr;
    rotatingplane::Workspace* workspace=nullptr;
    unsigned limbs=0;
};

// Sphere impulses include tangential contact velocity and angular inertia.
// A null second body is a kinematic Geometry or unit-boundary contact.
inline void solidImpulse(ParticleRigidBody& a,ParticleRigidBody* b,const SolidVector& normal,
                         const SolidVector& surface,const ParticleParams& p)
{
    const float inverseMass=1/p.bodyMass,r=p.bodyRadius;
    SolidVector armA {-r*normal[0],-r*normal[1],-r*normal[2]};
    SolidVector armB {r*normal[0],r*normal[1],r*normal[2]};
    auto va=solidVelocity(a),vb=b ? solidVelocity(*b):surface;
    const auto spinA=solidCross(a.angular,armA),spinB=b ? solidCross(b->angular,armB):SolidVector{};
    SolidVector relative;
    for (int axis=0;axis<3;++axis) relative[axis]=va[axis]+spinA[axis]-vb[axis]-spinB[axis];
    const float speed=solidDot(relative,normal);
    if (speed>=0) return;
    const float impulse=-(1+p.restitution)*speed/(inverseMass*(b ? 2:1));
    SolidVector tangent;
    for (int axis=0;axis<3;++axis) tangent[axis]=relative[axis]-speed*normal[axis];
    const float tangentLength=std::sqrt(solidDot(tangent,tangent));
    const float friction=std::min(p.friction*impulse,tangentLength/(inverseMass*(b ? 7:3.5f)));
    SolidVector push;
    for (int axis=0;axis<3;++axis)
        push[axis]=impulse*normal[axis]-(tangentLength>1.0e-7f ? friction*tangent[axis]/tangentLength:0);
    const float inverseInertia=2.5f*inverseMass/(r*r);
    const auto torqueA=solidCross(armA,push),torqueB=solidCross(armB,push);
    SolidVector acceleration;
    for (int axis=0;axis<3;++axis) {
        acceleration[axis]=push[axis]*inverseMass;
        a.angular[axis]=std::clamp(a.angular[axis]+torqueA[axis]*inverseInertia,-32.0f,32.0f);
        if (b) b->angular[axis]=std::clamp(b->angular[axis]-torqueB[axis]*inverseInertia,-32.0f,32.0f);
    }
    solidVelocityAdd(a,acceleration);
    if (b) { for (auto& v:acceleration) v=-v; solidVelocityAdd(*b,acceleration); }
}

inline void solidSphereContact(ParticleRigidBody& a,ParticleRigidBody* b,
                              const SolidVector& center,float otherRadius,
                              const SolidVector& surface,const ParticleParams& p)
{
    const auto position=solidPosition(a);
    SolidVector normal;
    for (int axis=0;axis<3;++axis) normal[axis]=position[axis]-center[axis];
    const float distance=std::sqrt(solidDot(normal,normal)),reach=p.bodyRadius+otherRadius;
    if (distance>=reach) return;
    if (distance>1.0e-7f) for (auto& value:normal) value/=distance;
    else normal={1,0,0};
    solidMove(a,normal,(reach-distance)*(b ? 0.5f:1));
    if (b) solidMove(*b,normal,-(reach-distance)*0.5f);
    solidImpulse(a,b,normal,surface,p);
}

inline void solidSphereSweep(ParticleRigidBody& a,ParticleRigidBody* b,
                             const SolidVector& fromA,const SolidVector& fromB,
                             const SolidVector& toB,float otherRadius,const SolidVector& surface,
                             const ParticleParams& p)
{
    const auto toA=solidPosition(a);
    SolidVector origin,travel;
    for (int axis=0;axis<3;++axis) {
        origin[axis]=fromA[axis]-fromB[axis];
        travel[axis]=toA[axis]-fromA[axis]-(toB[axis]-fromB[axis]);
    }
    const float radius=p.bodyRadius+otherRadius;
    const float aa=solidDot(travel,travel),bb=2*solidDot(origin,travel),cc=solidDot(origin,origin)-radius*radius;
    const float discriminant=bb*bb-4*aa*cc;
    if (cc<0 || aa<1.0e-12f || discriminant<0) return;
    const float hit=(-bb-std::sqrt(discriminant))/(2*aa);
    if (hit<0 || hit>1) return;
    SolidVector normal,positionA,positionB;
    for (int axis=0;axis<3;++axis) {
        normal[axis]=(origin[axis]+hit*travel[axis])/radius;
        positionB[axis]=b ? fromB[axis]+hit*(toB[axis]-fromB[axis]):toB[axis];
        positionA[axis]=positionB[axis]+normal[axis]*radius;
    }
    a.x=positionA[0]; a.y=positionA[1]; a.z=positionA[2];
    if (b) { b->x=positionB[0]; b->y=positionB[1]; b->z=positionB[2]; }
    solidImpulse(a,b,normal,surface,p);
}

// The interim guard bounds the entire replay, including every solver pass.
// It is a source work limit, not a measured device performance policy.
struct SolidTriangleWorkBudget
{
    static constexpr std::uint64_t maximum=100000000;
    static constexpr std::uint64_t queriesPerSweep=64;
    std::uint64_t used=0,limit=maximum;
    bool charge(std::uint64_t count=1)
    { if (used>limit || count>limit-used) return false; used+=count; return true; }
    bool preflight(std::uint64_t steps,std::uint64_t bodies,std::uint64_t vertices,std::uint64_t triangles,
                   std::uint64_t pointColliders=64,std::uint64_t queryCost=1) const
    {
        if (steps>1200+kParticleReplayMaxEvents || bodies>static_cast<std::uint64_t>(kParticleSolidMaxBodies) || vertices>256 || triangles>128 || pointColliders>64
            || queryCost==0 || queryCost>96) return false;
        const auto multiply=[](std::uint64_t a,std::uint64_t b,std::uint64_t& result)
        { if (b && a>maximum/b) return false; result=a*b; return true; };
        std::uint64_t samples=0,passes=0,queries=0;
        // Original motion, response projection, remaining motion, every point
        // correction and three wall corrections; motion guard and two queries.
        if (!multiply((6+2*pointColliders)*queriesPerSweep+3,triangles,queries)
            || !multiply(queries,queryCost,queries)) return false;
        const auto validation=2*(vertices+triangles);
        if (queries>maximum-validation) return false;
        const auto perPass=validation+queries;
        if (!multiply(steps+1,vertices+triangles,samples) || !multiply(steps,bodies,passes)
            || !multiply(passes,4,passes) || !multiply(passes,perPass,passes)) return false;
        return samples<=limit && passes<=limit-samples;
    }
    // Largest exact width n in [22,87] for the rotating coordinate-plane class
    // with W_old+steps*bodies*W(n,V,T,P) within this limit, where W_old is the
    // unchanged 96-feature fixed-axis bound above. Zero records the class as
    // unavailable; it never rejects an admitted fixed-axis request.
    unsigned rotatingWidth(std::uint64_t steps,std::uint64_t bodies,std::uint64_t vertices,
                           std::uint64_t triangles,std::uint64_t pointColliders) const
    {
        if (!preflight(steps,bodies,vertices,triangles,pointColliders,96)) return 0;
        const auto queries=((6+2*pointColliders)*queriesPerSweep+3)*triangles*96;
        const auto old=(steps+1)*(vertices+triangles)+steps*bodies*4*(2*(vertices+triangles)+queries);
        for (unsigned n=87;n>=22;--n) {
            const auto each=rotatingplane::ResponseCost::interval(n,vertices,triangles,pointColliders);
            const auto intervals=steps*bodies;
            if (each>maximum || (intervals && each>(limit-old)/intervals)) continue;
            return n;
        }
        return 0;
    }
};

namespace solidtriangle
{
using D3=std::array<double,3>;
struct Interval { double low=0,high=0; };
// Each elementary result widens one representable value. The normal-minimum
// padding also encloses underflow on targets that flush double subnormals.
inline double down(double v) { return std::nextafter(v,-std::numeric_limits<double>::infinity())-std::numeric_limits<double>::min(); }
inline double up(double v) { return std::nextafter(v,std::numeric_limits<double>::infinity())+std::numeric_limits<double>::min(); }
inline Interval exact(double v) { return {v,v}; }
inline Interval add(Interval a,Interval b) { return {down(a.low+b.low),up(a.high+b.high)}; }
inline Interval sub(Interval a,Interval b) { return {down(a.low-b.high),up(a.high-b.low)}; }
inline Interval mul(Interval a,Interval b)
{
    const std::array<double,4> values{a.low*b.low,a.low*b.high,a.high*b.low,a.high*b.high};
    return {down(*std::min_element(values.begin(),values.end())),up(*std::max_element(values.begin(),values.end()))};
}
inline bool arithmeticSupported()
{
#if defined(__FAST_MATH__) || defined(_M_FP_FAST) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
    return false;
#else
    return std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits==53
        && std::fegetround()==FE_TONEAREST;
#endif
}
inline D3 subtract(D3 a,D3 b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
inline double dot(D3 a,D3 b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline D3 cross(D3 a,D3 b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
inline D3 point(const SolidVector& p) { return {p[0],p[1],p[2]}; }
inline bool normalCoordinates(const SolidVector& p)
{ return std::all_of(p.begin(),p.end(),[](float v) { return std::isfinite(v) && (v==0 || std::abs(v)>=std::numeric_limits<float>::min()); }); }
using I3=std::array<Interval,3>;
inline I3 interpolate(const SolidVector& a,const SolidVector& b,Interval time)
{
    I3 result;
    for (int axis=0;axis<3;++axis) result[axis]=add(exact(a[axis]),mul(time,sub(exact(b[axis]),exact(a[axis]))));
    return result;
}
inline I3 interpolate(const SolidVector& a,const SolidVector& b,double time)
{ return interpolate(a,b,exact(time)); }
inline Interval motionTime(double time,double start,double end)
{
    if (start==end) return exact(start);
    if (start==0 && end==1) return exact(time);
    return add(exact(start),mul(exact(time),sub(exact(end),exact(start))));
}
inline D3 midpoint(const I3& p)
{ return {(p[0].low+p[0].high)*.5,(p[1].low+p[1].high)*.5,(p[2].low+p[2].high)*.5}; }
inline bool upperRoot(double squared,double& root)
{
    if (!std::isfinite(squared) || squared<0) return false;
    root=up(std::sqrt(up(squared+4*std::numeric_limits<double>::min())));
    // Verify the candidate by outward arithmetic rather than assuming a libm
    // sqrt accuracy. An unknown implementation refuses after bounded work.
    for (int attempt=0;attempt<8;++attempt) {
        if (std::isfinite(root) && root>=0 && mul(exact(root),exact(root)).low>=squared) return true;
        root=up(root);
    }
    return false;
}
inline bool normUpper(const I3& v,double& result)
{
    Interval squared{};
    for (const auto& axis:v) squared=add(squared,mul(axis,axis));
    return upperRoot(std::max(0.0,squared.high),result);
}
inline bool nondegenerate(const std::array<SolidVector,3>& triangle)
{
    I3 a,b,n;
    for (int axis=0;axis<3;++axis) {
        a[axis]=sub(exact(triangle[1][axis]),exact(triangle[0][axis]));
        b[axis]=sub(exact(triangle[2][axis]),exact(triangle[0][axis]));
    }
    for (int axis=0;axis<3;++axis) {
        n[axis]=sub(mul(a[(axis+1)%3],b[(axis+2)%3]),mul(a[(axis+2)%3],b[(axis+1)%3]));
        if (n[axis].low>0 || n[axis].high<0) return true;
    }
    return false;
}
// A single signed cross-product component over the entire linear interval
// certifies that no intermediate triangle collapses. Edge interpolation keeps
// common translation out of the interval widths.
inline bool nondegenerateMotion(const std::array<SolidVector,3>& old,const std::array<SolidVector,3>& current)
{
    I3 a,b;
    for (int axis=0;axis<3;++axis) {
        const auto edge=[&](int vertex) {
            const auto before=sub(exact(old[vertex][axis]),exact(old[0][axis]));
            const auto after=sub(exact(current[vertex][axis]),exact(current[0][axis]));
            return add(before,mul({0,1},sub(after,before)));
        };
        a[axis]=edge(1); b[axis]=edge(2);
    }
    for (int axis=0;axis<3;++axis) {
        const auto n=sub(mul(a[(axis+1)%3],b[(axis+2)%3]),mul(a[(axis+2)%3],b[(axis+1)%3]));
        if (n.low>0 || n.high<0) return true;
    }
    return false;
}
// Face/edge/vertex Voronoi regions. These barycentrics choose a candidate
// inside the triangle; the independent support bound certifies advancement.
inline std::array<double,3> closestWeights(D3 p,const std::array<D3,3>& v)
{
    const auto ab=subtract(v[1],v[0]),ac=subtract(v[2],v[0]),ap=subtract(p,v[0]);
    const double d1=dot(ab,ap),d2=dot(ac,ap);
    if (d1<=0 && d2<=0) return {1,0,0};
    const auto bp=subtract(p,v[1]); const double d3=dot(ab,bp),d4=dot(ac,bp);
    if (d3>=0 && d4<=d3) return {0,1,0};
    const double vc=d1*d4-d3*d2;
    if (vc<=0 && d1>=0 && d3<=0) { const double t=d1/(d1-d3); return {1-t,t,0}; }
    const auto cp=subtract(p,v[2]); const double d5=dot(ab,cp),d6=dot(ac,cp);
    if (d6>=0 && d5<=d6) return {0,0,1};
    const double vb=d5*d2-d1*d6;
    if (vb<=0 && d2>=0 && d6<=0) { const double t=d2/(d2-d6); return {1-t,0,t}; }
    const double va=d3*d6-d5*d4;
    if (va<=0 && d4-d3>=0 && d5-d6>=0) { const double t=(d4-d3)/((d4-d3)+(d5-d6)); return {0,1-t,t}; }
    const double inverse=1/(va+vb+vc),b=vb*inverse,c=vc*inverse;
    return {1-b-c,b,c};
}
struct Query
{
    double lower=0,upper=0,time=0,tolerance=0;
    D3 position{},closest{},normal{},boxClosest{};
    std::array<double,3> weights{};
};
inline bool query(const SolidVector& from,const SolidVector& to,
    const std::array<SolidVector,3>& old,const std::array<SolidVector,3>& current,double time,Query& result,Interval meshTime)
{
    result={}; result.time=time;
    const auto position=interpolate(from,to,time); std::array<I3,3> vertices;
    std::array<D3,3> approximate;
    for (int i=0;i<3;++i) { vertices[i]=interpolate(old[i],current[i],meshTime); approximate[i]=midpoint(vertices[i]); }
    result.position=midpoint(position);
    const auto raw=closestWeights(result.position,approximate);
    if (!std::all_of(raw.begin(),raw.end(),[](double v) { return std::isfinite(v); })) return false;
    // Nested interpolation guarantees convex weights even if the approximate
    // region calculation rounds a barycentric component just outside [0,1].
    const double v=std::clamp(raw[2],0.0,1.0),u=v==1 ? 0:std::clamp(raw[1]/(1-v),0.0,1.0);
    result.weights={(1-v)*(1-u),(1-v)*u,v};
    I3 closest,delta;
    for (int axis=0;axis<3;++axis) {
        const auto edge=add(vertices[0][axis],mul(exact(u),sub(vertices[1][axis],vertices[0][axis])));
        closest[axis]=add(edge,mul(exact(v),sub(vertices[2][axis],edge)));
        delta[axis]=sub(position[axis],closest[axis]);
    }
    result.closest=midpoint(closest); result.normal=subtract(result.position,result.closest);
    if (!normUpper(delta,result.upper)) return false;
    double normalLength=0; I3 exactNormal;
    for (int axis=0;axis<3;++axis) exactNormal[axis]=exact(result.normal[axis]);
    if (!normUpper(exactNormal,normalLength)) return false;
    if (normalLength>1.0e-150) {
        double separation=std::numeric_limits<double>::infinity();
        for (const auto& vertex:vertices) {
            Interval support{};
            for (int axis=0;axis<3;++axis) support=add(support,mul(sub(position[axis],vertex[axis]),exactNormal[axis]));
            separation=std::min(separation,support.low);
        }
        result.lower=std::max(0.0,down(separation/normalLength));
        for (auto& component:result.normal) component/=normalLength;
    } else {
        result.normal=cross(subtract(approximate[1],approximate[0]),subtract(approximate[2],approximate[0]));
        const double length=std::sqrt(dot(result.normal,result.normal));
        if (!std::isfinite(length) || length<=0) return false;
        for (auto& component:result.normal) component/=length;
        if (dot(result.normal,subtract(point(to),point(from)))>0) for (auto& component:result.normal) component=-component;
    }
    double magnitude=1;
    for (const auto& vertex:approximate) for (double component:vertex) magnitude=std::max(magnitude,std::abs(component));
    for (double component:result.position) magnitude=std::max(magnitude,std::abs(component));
    result.tolerance=8*std::numeric_limits<float>::epsilon()*magnitude;
    return std::isfinite(result.lower) && std::isfinite(result.upper) && result.lower<=result.upper;
}
inline bool query(const SolidVector& from,const SolidVector& to,
    const std::array<SolidVector,3>& old,const std::array<SolidVector,3>& current,double time,Query& result)
{ return query(from,to,old,current,time,result,exact(time)); }
// For fixed barycentrics, relative motion is a convex combination of the
// three relative vertex displacements. Its norm is <= their maximum norm.
// Thus distance can decrease by at most L*deltaTime on the normalized interval.
inline bool sweep(const SolidVector& from,const SolidVector& to,
    const std::array<SolidVector,3>& old,const std::array<SolidVector,3>& current,double reach,
    SolidTriangleWorkBudget& work,Query& contact,bool& hit,std::string& error,double meshStart=0,double meshEnd=1)
{
    hit=false;
    if (!std::isfinite(meshStart) || !std::isfinite(meshEnd) || meshStart<0 || meshEnd>1 || meshStart>meshEnd)
    { error="Triangle sweep has an invalid retained motion interval"; return false; }
    double speed=0;
    for (int vertex=0;vertex<3;++vertex) {
        I3 relative;
        for (int axis=0;axis<3;++axis) relative[axis]=sub(sub(exact(to[axis]),exact(from[axis])),
            mul(sub(exact(meshEnd),exact(meshStart)),sub(exact(current[vertex][axis]),exact(old[vertex][axis]))));
        double bound=0; if (!normUpper(relative,bound)) { error="Triangle sweep cannot certify its motion bound"; return false; }
        speed=std::max(speed,bound);
    }
    double time=0;
    for (std::uint64_t iteration=0;iteration<SolidTriangleWorkBudget::queriesPerSweep;++iteration) {
        if (!work.charge()) { error="Triangle contact replay exceeds 100M checked work units"; return false; }
        Query at;
        if (!query(from,to,old,current,time,at,motionTime(time,meshStart,meshEnd))) { error="Triangle contact distance is numerically uncertifiable"; return false; }
        // For a linear sphere path and linear vertices, endpoint support
        // against one fixed plane bounds support throughout the interval.
        // This also certifies remaining departure with fast tangential motion,
        // without asking conservative distance advancement to converge on it.
        if (time==0 && reach>at.tolerance && at.lower>=reach-at.tolerance) {
            double initial=std::numeric_limits<double>::infinity(),separation=initial,normalBound=0;
            I3 plane;
            for (int axis=0;axis<3;++axis) plane[axis]=exact(at.normal[axis]);
            if (!normUpper(plane,normalBound)) { error="Triangle departure plane is numerically uncertifiable"; return false; }
            for (int vertex=0;vertex<3;++vertex) {
                const auto before=interpolate(old[vertex],current[vertex],meshStart);
                const auto after=interpolate(old[vertex],current[vertex],meshEnd);
                Interval support{},start{};
                for (int axis=0;axis<3;++axis) {
                    support=add(support,mul(sub(exact(to[axis]),after[axis]),plane[axis]));
                    start=add(start,mul(sub(exact(from[axis]),before[axis]),plane[axis]));
                }
                initial=std::min(initial,start.low);
                separation=std::min(separation,support.low);
            }
            if (initial>=up((reach-at.tolerance)*normalBound)
                && separation>up((reach+at.tolerance)*normalBound)) return true;
        }
        if (at.upper<=reach+at.tolerance) { contact=at; hit=true; return true; }
        const double gap=down(at.lower-reach);
        if (!(gap>0) || !(speed>0)) { error="Triangle sweep cannot certify progress outside its contact tolerance"; return false; }
        const double safe=down(gap/speed);
        if (safe>=up(1-time)) return true;
        const double next=time+safe*.5;
        if (!(next>time) || next>1 || !std::isfinite(next) || up(next-time)>safe)
        { error="Triangle sweep cannot certify a representable next time"; return false; }
        time=next;
    }
    error="Triangle sweep exceeds 64 certified advancement queries; this motion is unsupported"; return false;
}
// Interval vertices of a linearly moving triangle at one retained mesh time.
// Static endpoints give exact vertices up to one outward rounding step.
inline std::array<I3,3> meshTriangle(const std::array<SolidVector,3>& old,const std::array<SolidVector,3>& current,Interval meshTime)
{
    std::array<I3,3> result;
    for (int i=0;i<3;++i) result[i]=interpolate(old[i],current[i],meshTime);
    return result;
}
inline std::array<I3,3> exactTriangle(const std::array<SolidVector,3>& triangle)
{
    std::array<I3,3> result;
    for (int i=0;i<3;++i) for (int axis=0;axis<3;++axis) result[i][axis]=exact(triangle[i][axis]);
    return result;
}
// A box query uses points inside the actual box hull and triangle, then
// certifies distance with their support planes. Candidate feature arithmetic
// chooses points only; it never certifies separation or safe advancement.
// Interval triangle vertices enclose the actual triangle, so every support
// bound and candidate distance below also holds for a moving triangle.
inline bool boxQuery(const SolidVector& from,const SolidVector& to,const SolidBoxProxy& box,
    const std::array<I3,3>& tri,double time,Query& result,SolidTriangleWorkBudget& work)
{
    result={}; result.time=time; result.upper=std::numeric_limits<double>::infinity();
    const auto center=interpolate(from,to,time);
    std::array<I3,8> vertices; std::array<D3,8> approximate;
    // Exact vertices have exact midpoints, so a static triangle keeps the
    // released candidate points bit for bit.
    std::array<D3,3> points;
    for (int i=0;i<3;++i) points[i]=midpoint(tri[i]);
    for (int vertex=0;vertex<8;++vertex) {
        vertices[vertex]=center;
        for (int edge=0;edge<3;++edge) for (int axis=0;axis<3;++axis)
            vertices[vertex][axis]=add(vertices[vertex][axis],mul(exact(box.axes[edge][axis]),
                exact((vertex&(1<<edge)) ? box.half[edge]:-box.half[edge])));
        approximate[vertex]=midpoint(vertices[vertex]);
    }
    const auto blend=[](const I3& a,const I3& b,double t) {
        I3 out; t=std::clamp(t,0.0,1.0);
        for (int axis=0;axis<3;++axis) out[axis]=add(a[axis],mul(exact(t),sub(b[axis],a[axis])));
        return out;
    };
    const auto trianglePoint=[&](const std::array<I3,3>& v,std::array<double,3> raw,
                                std::array<double,3>& weights) {
        const double t=std::clamp(raw[2],0.0,1.0),u=t==1 ? 0:std::clamp(raw[1]/(1-t),0.0,1.0);
        weights={(1-t)*(1-u),(1-t)*u,t};
        return blend(blend(v[0],v[1],u),v[2],t);
    };
    I3 bestBox{},bestTriangle{};
    const auto candidate=[&](const I3& a,const I3& b,const std::array<double,3>& weights) {
        if (!work.charge()) return false;
        I3 delta; for (int axis=0;axis<3;++axis) delta[axis]=sub(a[axis],b[axis]);
        double upper=0; if (!normUpper(delta,upper)) return false;
        const double tie=32*std::numeric_limits<double>::epsilon()*std::max(1.0,result.upper);
        if (!std::isfinite(result.upper) || upper<result.upper-tie)
        { result.upper=upper; bestBox=a; bestTriangle=b; result.weights=weights; }
        return true;
    };
    // Project the center's triangle candidate into the box first. Equal
    // face distances then retain a centered contact instead of an arbitrary
    // corner torque on a symmetric face impact.
    const auto a=point(box.axes[0]),b=point(box.axes[1]),c=point(box.axes[2]);
    const double determinant=dot(a,cross(b,c));
    if (!std::isfinite(determinant) || std::abs(determinant)<1.0e-12) return false;
    const auto inBox=[&](const I3& onTriangle,const std::array<double,3>& weights) {
        const auto offset=subtract(midpoint(onTriangle),midpoint(center));
        const std::array<double,3> local{dot(offset,cross(b,c))/determinant,
            dot(offset,cross(c,a))/determinant,dot(offset,cross(a,b))/determinant};
        std::array<double,3> t;
        for (int axis=0;axis<3;++axis) t[axis]=std::clamp((local[axis]/box.half[axis]+1)*.5,0.0,1.0);
        const auto low=blend(blend(vertices[0],vertices[1],t[0]),blend(vertices[2],vertices[3],t[0]),t[1]);
        const auto high=blend(blend(vertices[4],vertices[5],t[0]),blend(vertices[6],vertices[7],t[0]),t[1]);
        return candidate(blend(low,high,t[2]),onTriangle,weights);
    };
    std::array<double,3> weights;
    const auto centerTriangle=trianglePoint(tri,closestWeights(midpoint(center),points),weights);
    if (!inBox(centerTriangle,weights)) return false;
    for (int vertex=0;vertex<8;++vertex) {
        const auto onTriangle=trianglePoint(tri,closestWeights(approximate[vertex],points),weights);
        if (!candidate(vertices[vertex],onTriangle,weights)) return false;
    }
    // Triangles wholly inside a box have no boundary feature intersection.
    // Clamped coordinates still give points inside the actual convex hull.
    for (int vertex=0;vertex<3;++vertex) {
        weights={}; weights[vertex]=1;
        if (!inBox(tri[vertex],weights)) return false;
    }
    const auto segmentFace=[&](const I3& start,const I3& end,const std::array<I3,3>& face,
                              bool boxSegment,int triangleEdge) {
        const auto x=midpoint(start),y=midpoint(end);
        const std::array<D3,3> v{midpoint(face[0]),midpoint(face[1]),midpoint(face[2])};
        const auto normal=cross(subtract(v[1],v[0]),subtract(v[2],v[0]));
        const double speed=dot(normal,subtract(y,x));
        if (speed==0) return true;
        const double t=std::clamp(dot(normal,subtract(v[0],x))/speed,0.0,1.0);
        const auto onSegment=blend(start,end,t);
        std::array<double,3> weights;
        const auto onFace=trianglePoint(face,closestWeights(midpoint(onSegment),v),weights);
        if (boxSegment) return candidate(onSegment,onFace,weights);
        weights={}; weights[triangleEdge]=1-t; weights[(triangleEdge+1)%3]=t;
        return candidate(onFace,onSegment,weights);
    };
    for (int axis=0;axis<3;++axis) for (int start=0;start<8;++start) if (!(start&(1<<axis))) {
        const int end=start|(1<<axis);
        if (!segmentFace(vertices[start],vertices[end],tri,true,0)) return false;
        for (int edge=0;edge<3;++edge) {
            const auto u=subtract(approximate[end],approximate[start]);
            const auto v=subtract(points[(edge+1)%3],points[edge]);
            const auto w=subtract(approximate[start],points[edge]);
            const double uu=dot(u,u),vv=dot(v,v),uv=dot(u,v),uw=dot(u,w),vw=dot(v,w);
            const double denominator=uu*vv-uv*uv;
            double t=denominator>0 ? std::clamp((uv*vw-vv*uw)/denominator,0.0,1.0):0;
            const double q=std::clamp((uv*t+vw)/vv,0.0,1.0);
            t=std::clamp((uv*q-uw)/uu,0.0,1.0);
            std::array<double,3> weights{}; weights[edge]=1-q; weights[(edge+1)%3]=q;
            if (!candidate(blend(vertices[start],vertices[end],t),blend(tri[edge],tri[(edge+1)%3],q),weights)) return false;
        }
    }
    for (int axis=0;axis<3;++axis) for (int side=0;side<2;++side) {
        const int origin=side<<axis,one=origin|(1<<((axis+1)%3)),two=origin|(1<<((axis+2)%3)),both=one|two;
        for (const auto& face:std::array<std::array<int,3>,2>{{{origin,one,both},{origin,both,two}}})
            for (int edge=0;edge<3;++edge)
                if (!segmentFace(tri[edge],tri[(edge+1)%3],
                    {vertices[face[0]],vertices[face[1]],vertices[face[2]]},false,edge)) return false;
    }
    result.position=midpoint(center); result.closest=midpoint(bestTriangle); result.boxClosest=midpoint(bestBox);
    const auto delta=subtract(midpoint(bestBox),result.closest);
    double exit=std::numeric_limits<double>::infinity(); D3 exitNormal{},separatingNormal{};
    const auto support=[&](D3 normal,bool chooseExit) {
        I3 direction; for (int axis=0;axis<3;++axis) direction[axis]=exact(normal[axis]);
        double length=0; if (!normUpper(direction,length)) return false;
        if (length<1.0e-150) return true;
        for (int sign:{1,-1}) {
            double low=std::numeric_limits<double>::infinity(),high=-low;
            for (const auto& vertex:vertices) {
                Interval projection{};
                for (int axis=0;axis<3;++axis) projection=add(projection,mul(vertex[axis],exact(sign*normal[axis])));
                low=std::min(low,projection.low);
            }
            for (const auto& vertex:tri) {
                Interval projection{};
                for (int axis=0;axis<3;++axis) projection=add(projection,mul(vertex[axis],exact(sign*normal[axis])));
                high=std::max(high,projection.high);
            }
            const double separation=down(down(low-high)/length);
            if (separation>0) {
                const double tie=32*std::numeric_limits<double>::epsilon()*std::max(1.0,result.lower);
                if (result.lower==0 || separation>result.lower+tie)
                    for (int axis=0;axis<3;++axis) separatingNormal[axis]=sign*normal[axis]/length;
                result.lower=std::max(result.lower,separation);
            }
            if (chooseExit && -separation<exit) {
                exit=-separation;
                for (int axis=0;axis<3;++axis) exitNormal[axis]=sign*normal[axis]/length;
            }
        }
        return true;
    };
    for (int axis=0;axis<3;++axis) if (!support(cross(point(box.axes[(axis+1)%3]),point(box.axes[(axis+2)%3])),true)) return false;
    if (!support(cross(subtract(points[1],points[0]),subtract(points[2],points[0])),true)) return false;
    for (int axis=0;axis<3;++axis) for (int edge=0;edge<3;++edge)
        if (!support(cross(point(box.axes[axis]),subtract(points[(edge+1)%3],points[edge])),true)) return false;
    if (!support(delta,false)) return false;
    // Prefer a certified face plane when its distance bound ties a candidate
    // direction within double roundoff. Centered face impacts then keep the
    // actual face normal rather than an interpolation-noise torque.
    result.normal=result.lower>0 ? separatingNormal:exitNormal;
    double magnitude=1;
    for (const auto& vertex:approximate) for (double component:vertex) magnitude=std::max(magnitude,std::abs(component));
    for (const auto& vertex:points) for (double component:vertex) magnitude=std::max(magnitude,std::abs(component));
    result.tolerance=8*std::numeric_limits<float>::epsilon()*magnitude;
    return std::isfinite(result.upper) && std::isfinite(result.lower) && result.lower<=result.upper;
}
inline bool boxQuery(const SolidVector& from,const SolidVector& to,const SolidBoxProxy& box,
    const std::array<SolidVector,3>& triangle,double time,Query& result,SolidTriangleWorkBudget& work)
{ return boxQuery(from,to,box,exactTriangle(triangle),time,result,work); }
inline bool boxSeparation(const I3& center,const SolidBoxProxy& box,
    const std::array<I3,3>& triangle,D3 normal,double& separation,double& length)
{
    I3 direction; for (int axis=0;axis<3;++axis) direction[axis]=exact(normal[axis]);
    if (!normUpper(direction,length) || !(length>0)) return false;
    Interval projection{},radius{};
    for (int axis=0;axis<3;++axis) projection=add(projection,mul(center[axis],direction[axis]));
    for (int edge=0;edge<3;++edge) {
        Interval facing{};
        for (int axis=0;axis<3;++axis) facing=add(facing,mul(exact(box.axes[edge][axis]),direction[axis]));
        const double absolute=std::max(std::abs(facing.low),std::abs(facing.high));
        radius=add(radius,mul(exact(box.half[edge]),exact(absolute)));
    }
    double high=-std::numeric_limits<double>::infinity();
    for (const auto& vertex:triangle) {
        Interval value{};
        for (int axis=0;axis<3;++axis) value=add(value,mul(vertex[axis],direction[axis]));
        high=std::max(high,value.high);
    }
    separation=sub(sub(projection,radius),exact(high)).low;
    return std::isfinite(separation);
}
inline bool boxSeparation(const I3& center,const SolidBoxProxy& box,
    const std::array<SolidVector,3>& triangle,D3 normal,double& separation,double& length)
{ return boxSeparation(center,box,exactTriangle(triangle),normal,separation,length); }
inline bool boxSweep(const SolidVector& from,const SolidVector& to,const SolidBoxProxy& box,
    const std::array<SolidVector,3>& triangle,double reach,SolidTriangleWorkBudget& work,
    Query& contact,bool& hit,std::string& error)
{
    hit=false;
    I3 travel; for (int axis=0;axis<3;++axis) travel[axis]=sub(exact(to[axis]),exact(from[axis]));
    double speed=0;
    if (!normUpper(travel,speed)) { error="Box triangle sweep cannot certify its motion bound"; return false; }
    double time=0;
    for (std::uint64_t iteration=0;iteration<SolidTriangleWorkBudget::queriesPerSweep;++iteration) {
        Query at;
        if (!boxQuery(from,to,box,triangle,time,at,work))
        { error=work.used>=work.limit ? "Triangle contact replay exceeds 100M checked work units":"Box triangle distance is numerically uncertifiable"; return false; }
        if (time==0 && at.lower>=reach-at.tolerance) {
            double initial=0,final=0,length=0;
            if (!boxSeparation(interpolate(from,to,0),box,triangle,at.normal,initial,length)
                || !boxSeparation(interpolate(from,to,1),box,triangle,at.normal,final,length))
            { error="Box triangle departure plane is numerically uncertifiable"; return false; }
            if (initial>=up((reach-at.tolerance)*length) && final>up((reach+at.tolerance)*length)) return true;
        }
        if (at.upper<=reach+at.tolerance) { contact=at; hit=true; return true; }
        const double gap=down(at.lower-reach);
        if (!(gap>0)) { error="Box triangle sweep cannot certify progress outside its contact tolerance"; return false; }
        if (!(speed>0)) return true;
        const double safe=down(gap/speed);
        if (safe>=up(1-time)) return true;
        const double next=time+safe*.5;
        if (!(next>time) || next>1 || !std::isfinite(next) || up(next-time)>safe)
        { error="Box triangle sweep cannot certify a representable next time"; return false; }
        time=next;
    }
    error="Box triangle sweep exceeds 64 certified advancement queries; this motion is unsupported"; return false;
}
// A fixed-orientation box against linearly moving triangle vertices. For
// fixed box and triangle feature points, relative motion is a convex
// combination of the three relative vertex displacements, so the sphere
// sweep's bound also limits how fast box distance can fall. Separation along
// one fixed plane is linear minus a maximum of linear terms, hence concave in
// time: endpoint departure bounds the whole retained interval.
inline bool movingBoxSweep(const SolidVector& from,const SolidVector& to,const SolidBoxProxy& box,
    const std::array<SolidVector,3>& old,const std::array<SolidVector,3>& current,double reach,
    SolidTriangleWorkBudget& work,Query& contact,bool& hit,std::string& error,double meshStart=0,double meshEnd=1)
{
    hit=false;
    if (!std::isfinite(meshStart) || !std::isfinite(meshEnd) || meshStart<0 || meshEnd>1 || meshStart>meshEnd)
    { error="Box triangle sweep has an invalid retained motion interval"; return false; }
    double speed=0;
    for (int vertex=0;vertex<3;++vertex) {
        I3 relative;
        for (int axis=0;axis<3;++axis) relative[axis]=sub(sub(exact(to[axis]),exact(from[axis])),
            mul(sub(exact(meshEnd),exact(meshStart)),sub(exact(current[vertex][axis]),exact(old[vertex][axis]))));
        double bound=0; if (!normUpper(relative,bound)) { error="Box triangle sweep cannot certify its motion bound"; return false; }
        speed=std::max(speed,bound);
    }
    double time=0;
    for (std::uint64_t iteration=0;iteration<SolidTriangleWorkBudget::queriesPerSweep;++iteration) {
        Query at;
        if (!boxQuery(from,to,box,meshTriangle(old,current,motionTime(time,meshStart,meshEnd)),time,at,work))
        { error=work.used>=work.limit ? "Triangle contact replay exceeds 100M checked work units":"Box triangle distance is numerically uncertifiable"; return false; }
        if (time==0 && at.lower>=reach-at.tolerance) {
            double initial=0,final=0,length=0;
            if (!boxSeparation(interpolate(from,to,0),box,meshTriangle(old,current,exact(meshStart)),at.normal,initial,length)
                || !boxSeparation(interpolate(from,to,1),box,meshTriangle(old,current,exact(meshEnd)),at.normal,final,length))
            { error="Box triangle departure plane is numerically uncertifiable"; return false; }
            if (initial>=up((reach-at.tolerance)*length) && final>up((reach+at.tolerance)*length)) return true;
        }
        if (at.upper<=reach+at.tolerance) { contact=at; hit=true; return true; }
        const double gap=down(at.lower-reach);
        if (!(gap>0)) { error="Box triangle sweep cannot certify progress outside its contact tolerance"; return false; }
        if (!(speed>0)) return true;
        const double safe=down(gap/speed);
        if (safe>=up(1-time)) return true;
        const double next=time+safe*.5;
        if (!(next>time) || next>1 || !std::isfinite(next) || up(next-time)>safe)
        { error="Box triangle sweep cannot certify a representable next time"; return false; }
        time=next;
    }
    error="Box triangle sweep exceeds 64 certified advancement queries; this motion is unsupported"; return false;
}

}

inline bool validateSolidTriangleFrame(const ParticleGeometryFrame& frame,SolidTriangleWorkBudget& work,std::string& error)
{
    if (!frame.meshIdentity) {
        if (frame.meshVertexCount==0 && frame.meshTriangleCount==0) return true;
        error="Triangle vertices require their stable mesh owner"; return false;
    }
    if (!solidtriangle::arithmeticSupported()) { error="Triangle contacts require IEEE binary64 nearest-rounding arithmetic without fast math"; return false; }
    if (frame.meshVertexCount==0 || frame.meshVertexCount>256 || frame.meshTriangleCount==0 || frame.meshTriangleCount>128)
    { error="Triangle surface exceeds 256 vertices or 128 triangles"; return false; }
    std::array<std::uint64_t,256> identities;
    for (std::uint32_t vertex=0;vertex<frame.meshVertexCount;++vertex) {
        if (!work.charge()) { error="Triangle contact replay exceeds 100M checked work units"; return false; }
        if (!frame.meshVertexIdentities[vertex]
            || !std::all_of(frame.meshVertices[vertex].begin(),frame.meshVertices[vertex].end(),[](float v) {
                return std::isfinite(v) && std::abs(v)<=32 && (v==0 || std::abs(v)>=std::numeric_limits<float>::min()); }))
        { error="Triangle surface lost its finite vertices or stable identities"; return false; }
        identities[vertex]=frame.meshVertexIdentities[vertex];
    }
    // Sort only this stack copy; source vertex and triangle order stays intact.
    const auto end=identities.begin()+frame.meshVertexCount;
    std::make_heap(identities.begin(),end);
    std::sort_heap(identities.begin(),end);
    if (std::adjacent_find(identities.begin(),end)!=end)
    { error="Triangle surface lost its finite vertices or stable identities"; return false; }
    for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
        if (!work.charge()) { error="Triangle contact replay exceeds 100M checked work units"; return false; }
        std::array<SolidVector,3> vertices;
        for (int corner=0;corner<3;++corner) {
            const auto index=frame.meshTriangles[triangle][corner];
            if (index>=frame.meshVertexCount) { error="Triangle surface has an invalid vertex index"; return false; }
            vertices[corner]=frame.meshVertices[index];
        }
        if (!solidtriangle::nondegenerate(vertices)) { error="Degenerate or numerically uncertain triangle surfaces are unsupported"; return false; }
    }
    return true;
}

// A box spinning only about the normal of a static coordinate-plane mesh has
// normal support h_k at every pose, so its linear normal clearance bounds the
// whole rotational sweep. Near the plane the exact certificate supplies the
// first root, a finite triangle witness and the response. Other spins refuse.
inline bool solidRotatingBoxTriangleContact(ParticleRigidBody& candidate,const ParticleRigidBody& previous,
    const ParticleGeometryFrame& frame,const ParticleParams& p,float dt,const SolidVector& half,
    const SolidRotatingSweep* rotating,SolidTriangleWorkBudget& budget,std::string& error,
    SolidTriangleContactImpulse* impulse)
{
    // Direct calls without an issued view, and requests whose preflight
    // admitted no exact width, keep the released changing-axis refusal. This
    // rotating refusal itself charges nothing, like the released one. The
    // whole call still charges 2(V+T): validateSolidTriangleFrame runs on the
    // frame and the old frame before dispatch. The contract prices that as
    // the 2*(V+T) term of W_old (contract line 119).
    if (!rotating || !rotating->view || !rotating->workspace || !rotatingplane::Cost::supported(rotating->limbs))
    { error="Rotating box triangle sweeps are unsupported; the replay interval is refused"; return false; }
    std::array<float,4> raw {};
    SolidVector scheduledAngular {};
    if (!rotating->view->consume(candidate,previous,dt,raw,scheduledAngular))
    { error="Rotating box triangle sweep lost its replay rotation provenance; the replay interval is refused"; return false; }
    // These class checks charge nothing. The first exact phase's scalar term
    // (64 class/provenance checks plus 3V plane and exponent visits) prices
    // them, and preflight already admitted that phase for every interval.
    const auto onPlane=[&](unsigned axis) {
        for (std::uint32_t vertex=1;vertex<frame.meshVertexCount;++vertex)
            if (!rotatingplane::sameFloat(frame.meshVertices[vertex][axis],frame.meshVertices[0][axis])) return false;
        return true;
    };
    unsigned k=0;
    while (k<3 && !onPlane(k)) ++k;
    const auto aboutNormal=[&](const auto& components) {
        for (unsigned axis=0;axis<3;++axis) if (axis!=k && components[axis]!=0) return false;
        return true;
    };
    if (k==3 || !aboutNormal(previous.orientation) || !aboutNormal(candidate.orientation)
        || !aboutNormal(raw) || !aboutNormal(scheduledAngular))
    { error="Rotating box triangle sweeps require a spin about one coordinate-plane mesh normal; the replay interval is refused"; return false; }
    const unsigned i=(k+1)%3,j=(k+2)%3;
    const auto from=solidPosition(previous),to=solidPosition(candidate);
    const float plane=frame.meshVertices[0][k];
    const int side=from[k]<plane ? -1:1;
    const auto clearance=[&](float center) {
        auto offset=solidtriangle::sub(solidtriangle::exact(center),solidtriangle::exact(plane));
        if (side<0) offset={-offset.high,-offset.low};
        return solidtriangle::sub(solidtriangle::sub(offset,solidtriangle::exact(half[k])),
            solidtriangle::exact(p.colliderThickness)).low;
    };
    if (clearance(from[k])>0 && clearance(to[k])>0) return true;
    const auto before=solidBox(previous,half),after=solidBox(candidate,half);
    const rotatingplane::Input input{from,to,half,p.colliderThickness,dt,plane,
        previous.orientation[k],previous.orientation[3],raw[k],raw[3],
        before.axes[i][i],before.axes[i][j],after.axes[i][i],after.axes[i][j],k};
    // The width is fixed per request: the largest one preflight admitted.
    // Every source coefficient and all six produced response floats must sit
    // on its grid; the certificate and response refuse atomically otherwise.
    const unsigned n=rotating->limbs;
    static_assert(sizeof(ParticleGeometryFrame::meshVertices)==256*3*sizeof(float)
        && sizeof(ParticleGeometryFrame::meshTriangles)==128*3*sizeof(std::uint32_t));
    const rotatingplane::Mesh mesh{frame.meshVertices[0].data(),frame.meshTriangles[0].data(),
        frame.meshVertexCount,frame.meshTriangleCount};
    const rotatingplane::ResponseInput state{solidVelocity(candidate),candidate.angular,scheduledAngular,
        p.bodyMass,p.restitution,p.friction};
    // The exact phases charge the shared replay budget before their operations.
    struct Work {
        SolidTriangleWorkBudget& budget;
        bool exhausted=false;
        bool charge(std::uint64_t count) { exhausted=!budget.charge(count); return !exhausted; }
    } work{budget};
    rotatingplane::Response result;
    if (!rotatingplane::response(input,mesh,state,n,*rotating->workspace,work,result)) {
        error=work.exhausted ? "Triangle contact replay exceeds 100M checked work units"
            :"Rotating box triangle sweep cannot certify its exact plane contact; the replay interval is refused";
        return false;
    }
    if (result.kind!=rotatingplane::Kind::impact) return true;
    // The certified remainder endpoint replaces the scheduled center. The
    // published orientation stays the executed solidRotate pose.
    candidate.x=result.center[0]; candidate.y=result.center[1]; candidate.z=result.center[2];
    candidate.vx=result.velocity[0]; candidate.vy=result.velocity[1]; candidate.vz=result.velocity[2];
    candidate.angular=result.angular;
    if (impulse) {
        impulse->impulse=result.impulse;
        impulse->valid=std::any_of(impulse->impulse.begin(),impulse->impulse.end(),[](double value) { return value!=0; });
        if (impulse->valid) {
            impulse->bodyIdentity=candidate.identity; impulse->bodyGeometryIdentity=candidate.geometryIdentity;
            impulse->meshIdentity=frame.meshIdentity; impulse->triangle=result.triangle;
            for (int corner=0;corner<3;++corner)
                impulse->vertexIdentities[corner]=frame.meshVertexIdentities[frame.meshTriangles[result.triangle][corner]];
            impulse->weights=result.weights; impulse->point=result.pointOnTriangle; impulse->time=result.time;
        }
    }
    return true;
}

// A fixed-orientation box against an animated mesh. Sampled endpoint vertices
// define linear motion across the replay interval. The response follows the
// moving sphere path: push out at the impact mesh time, add the weighted
// surface velocity to the box impulse, then certify the remaining motion
// against the moving mesh and the final pose against the current mesh.
// Per triangle this spends one motion guard, three sweeps and two queries,
// inside the preflight's fixed-axis box bound.
inline bool solidMovingBoxTriangleContact(ParticleRigidBody& candidate,const ParticleRigidBody& previous,
    const ParticleGeometryFrame& frame,const ParticleGeometryFrame& old,const ParticleParams& p,
    const SolidBoxProxy& box,SolidTriangleWorkBudget& budget,std::string& error,
    SolidTriangleContactImpulse* impulse,float dt)
{
    const auto from=solidPosition(previous),to=solidPosition(candidate);
    if (!solidtriangle::normalCoordinates(from) || !solidtriangle::normalCoordinates(to))
    { error="Box triangle contacts require finite normal-range coordinates"; return false; }
    const auto endpoints=[&](std::uint32_t triangle,std::array<SolidVector,3>& before,std::array<SolidVector,3>& current) {
        bool moving=false;
        for (int corner=0;corner<3;++corner) {
            const auto index=frame.meshTriangles[triangle][corner];
            before[corner]=old.meshVertices[index]; current[corner]=frame.meshVertices[index];
            moving=moving || before[corner]!=current[corner];
        }
        return moving;
    };
    bool found=false,firstMoving=false; std::uint32_t firstTriangle=0; solidtriangle::Query first;
    const double reach=p.colliderThickness;
    for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
        std::array<SolidVector,3> before,current;
        const bool moving=endpoints(triangle,before,current);
        if (moving) {
            if (!budget.charge()) { error="Triangle contact replay exceeds 100M checked work units"; return false; }
            if (!solidtriangle::nondegenerateMotion(before,current))
            { error="Moving triangle may degenerate during its retained linear interval; this frame is unsupported"; return false; }
        }
        solidtriangle::Query contact; bool hit=false;
        if (!solidtriangle::movingBoxSweep(from,to,box,before,current,reach,budget,contact,hit,error)) return false;
        if (hit && (!found || contact.time<first.time)) { found=true; first=contact; firstTriangle=triangle; firstMoving=moving; }
    }
    if (!found) return true;
    candidate.x=static_cast<float>(first.position[0]);
    candidate.y=static_cast<float>(first.position[1]);
    candidate.z=static_cast<float>(first.position[2]);
    const auto responseFrom=solidPosition(candidate);
    if (!solidtriangle::normalCoordinates(responseFrom))
    { error="Box triangle response requires finite normal-range coordinates"; return false; }
    SolidVector arm;
    for (int axis=0;axis<3;++axis) arm[axis]=static_cast<float>(first.boxClosest[axis])-responseFrom[axis];
    SolidVector normal; for (int axis=0;axis<3;++axis) normal[axis]=static_cast<float>(first.normal[axis]);
    // Push out against the impact pose, reusing the original vertex endpoints
    // so no rounded intermediate mesh becomes authority.
    const auto impactTime=solidtriangle::exact(first.time);
    std::array<SolidVector,3> firstBefore,firstCurrent;
    endpoints(firstTriangle,firstBefore,firstCurrent);
    double separation=0,length=0;
    if (!solidtriangle::boxSeparation(solidtriangle::interpolate(responseFrom,responseFrom,0),box,
        solidtriangle::meshTriangle(firstBefore,firstCurrent,impactTime),solidtriangle::point(normal),separation,length))
    { error="Box triangle response plane is numerically uncertifiable"; return false; }
    solidtriangle::Interval squared{};
    for (float value:normal) squared=solidtriangle::add(squared,solidtriangle::mul(solidtriangle::exact(value),solidtriangle::exact(value)));
    if (!(squared.low>0)) { error="Box triangle response has no certified normal"; return false; }
    const double distance=solidtriangle::up((solidtriangle::up((reach+2*first.tolerance)*length)-separation)/squared.low);
    if (!std::isfinite(distance)) { error="Box triangle response distance is not finite"; return false; }
    candidate.x=static_cast<float>(double(candidate.x)+normal[0]*distance);
    candidate.y=static_cast<float>(double(candidate.y)+normal[1]*distance);
    candidate.z=static_cast<float>(double(candidate.z)+normal[2]*distance);
    if (!solidtriangle::normalCoordinates(solidPosition(candidate)))
    { error="Box triangle response requires finite normal-range coordinates"; return false; }
    for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
        std::array<SolidVector,3> before,current;
        endpoints(triangle,before,current);
        if (triangle!=firstTriangle) {
            solidtriangle::Query projected; bool hit=false;
            if (!solidtriangle::movingBoxSweep(responseFrom,solidPosition(candidate),box,before,current,reach,
                budget,projected,hit,error,first.time,first.time)) return false;
            if (hit) { error="Box triangle response projection may cross another surface; this frame is unsupported"; return false; }
        }
        solidtriangle::Query final;
        if (!solidtriangle::boxQuery(solidPosition(candidate),solidPosition(candidate),box,
                solidtriangle::meshTriangle(before,current,impactTime),0,final,budget)
            || final.lower<reach-final.tolerance)
        { error=budget.used>=budget.limit ? "Triangle contact replay exceeds 100M checked work units":
            "Box triangle response cannot certify separation from every surface"; return false; }
    }
    SolidVector surface{};
    if (firstMoving) for (int axis=0;axis<3;++axis) {
        double velocity=0;
        for (int corner=0;corner<3;++corner)
            velocity+=first.weights[corner]*(double(firstCurrent[corner][axis])-firstBefore[corner][axis])/dt;
        if (!std::isfinite(velocity)) { error="Moving triangle surface velocity is not finite"; return false; }
        surface[axis]=static_cast<float>(std::clamp(velocity,-32.0,32.0));
    }
    const auto velocityBefore=solidVelocity(candidate);
    solidBoxSurfaceImpulse(candidate,box,arm,normal,surface,p);
    // The impulse changes spin for the next interval only; this interval
    // keeps the scheduled orientation, so the same box proxy stays exact.
    const auto remainingFrom=solidPosition(candidate),velocity=solidVelocity(candidate);
    const double remaining=double(dt)*(1-first.time);
    candidate.x=static_cast<float>(double(remainingFrom[0])+remaining*velocity[0]);
    candidate.y=static_cast<float>(double(remainingFrom[1])+remaining*velocity[1]);
    candidate.z=static_cast<float>(double(remainingFrom[2])+remaining*velocity[2]);
    if (!solidtriangle::normalCoordinates(solidPosition(candidate)))
    { error="Moving triangle response requires finite normal-range coordinates"; return false; }
    for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
        std::array<SolidVector,3> before,current;
        endpoints(triangle,before,current);
        solidtriangle::Query remainingContact; bool hit=false;
        if (!solidtriangle::movingBoxSweep(remainingFrom,solidPosition(candidate),box,before,current,reach,budget,
            remainingContact,hit,error,first.time,1)) return false;
        if (hit) { error="Moving triangle remaining response may cross a surface; this frame is unsupported"; return false; }
        solidtriangle::Query final;
        if (!solidtriangle::boxQuery(solidPosition(candidate),solidPosition(candidate),box,current,0,final,budget)
            || final.lower<reach-final.tolerance)
        { error=budget.used>=budget.limit ? "Triangle contact replay exceeds 100M checked work units":
            "Moving triangle response cannot certify final separation from every surface"; return false; }
    }
    if (impulse) {
        const auto velocityAfter=solidVelocity(candidate);
        for (int axis=0;axis<3;++axis) impulse->impulse[axis]=double(p.bodyMass)*(double(velocityAfter[axis])-velocityBefore[axis]);
        impulse->valid=std::any_of(impulse->impulse.begin(),impulse->impulse.end(),[](double value) { return value!=0; });
        if (impulse->valid) {
            impulse->bodyIdentity=candidate.identity; impulse->bodyGeometryIdentity=candidate.geometryIdentity;
            impulse->meshIdentity=frame.meshIdentity; impulse->triangle=firstTriangle;
            for (int corner=0;corner<3;++corner)
                impulse->vertexIdentities[corner]=frame.meshVertexIdentities[frame.meshTriangles[firstTriangle][corner]];
            impulse->weights=first.weights; impulse->point=first.closest; impulse->time=first.time;
        }
    }
    return true;
}

inline bool solidBoxTriangleContact(ParticleRigidBody& candidate,const ParticleRigidBody& previous,
    const ParticleGeometryFrame& frame,const ParticleGeometryFrame& old,const ParticleParams& p,
    bool sweep,bool correctionSweep,const SolidVector& half,SolidTriangleWorkBudget& budget,std::string& error,
    SolidTriangleContactImpulse* impulse,float dt,const SolidRotatingSweep* rotating)
{
    const auto box=solidBox(candidate,half);
    if (!std::all_of(half.begin(),half.end(),[](float value) { return std::isfinite(value) && value>1.0e-6f; })
        || !std::all_of(box.axes.begin(),box.axes.end(),[](const SolidVector& axis) { return solidtriangle::normalCoordinates(axis); }))
    { error="Box triangle contacts require finite nonzero box bounds and orientation"; return false; }
    bool movingMesh=false;
    for (std::uint32_t vertex=0;vertex<frame.meshVertexCount;++vertex)
        movingMesh=movingMesh || frame.meshVertices[vertex]!=old.meshVertices[vertex];
    // Compare the actual proxy axes, so q and -q retain the same pose. Only a
    // captured spin about a coordinate-plane mesh normal has a certified sweep,
    // and that class certifies a static mesh only.
    const bool spinning=sweep && solidBox(previous,half).axes!=box.axes;
    if (spinning && movingMesh)
    { error="Rotating box contacts against a moving triangle surface are unsupported; the replay interval is refused"; return false; }
    if (spinning)
        return solidRotatingBoxTriangleContact(candidate,previous,frame,p,dt,half,rotating,budget,error,impulse);
    // Correction passes check the current mesh pose, like the sphere path.
    if (sweep && movingMesh)
        return solidMovingBoxTriangleContact(candidate,previous,frame,old,p,box,budget,error,impulse,dt);
    const auto from=sweep || correctionSweep ? solidPosition(previous):solidPosition(candidate),to=solidPosition(candidate);
    if (!solidtriangle::normalCoordinates(from) || !solidtriangle::normalCoordinates(to))
    { error="Box triangle contacts require finite normal-range coordinates"; return false; }
    const auto vertices=[&](std::uint32_t triangle) {
        std::array<SolidVector,3> result;
        for (int corner=0;corner<3;++corner) result[corner]=frame.meshVertices[frame.meshTriangles[triangle][corner]];
        return result;
    };
    bool found=false; std::uint32_t firstTriangle=0; solidtriangle::Query first;
    const double reach=p.colliderThickness;
    for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
        solidtriangle::Query contact; bool hit=false;
        if (!solidtriangle::boxSweep(from,to,box,vertices(triangle),reach,budget,contact,hit,error)) return false;
        if (hit && (!found || contact.time<first.time)) { found=true; first=contact; firstTriangle=triangle; }
    }
    if (!found) return true;
    candidate.x=static_cast<float>(first.position[0]);
    candidate.y=static_cast<float>(first.position[1]);
    candidate.z=static_cast<float>(first.position[2]);
    const auto responseFrom=solidPosition(candidate);
    // Retain the actual box contact arm before projection. Collider reach
    // and response padding move the body without enlarging its inertia arm.
    // Convert contact coordinates to float before subtracting the float
    // center, so centered face impacts retain the released zero-spin result.
    SolidVector arm;
    for (int axis=0;axis<3;++axis) arm[axis]=static_cast<float>(first.boxClosest[axis])-responseFrom[axis];
    SolidVector normal; for (int axis=0;axis<3;++axis) normal[axis]=static_cast<float>(first.normal[axis]);
    double separation=0,length=0;
    if (!solidtriangle::boxSeparation(solidtriangle::interpolate(responseFrom,responseFrom,0),box,
        vertices(firstTriangle),solidtriangle::point(normal),separation,length))
    { error="Box triangle response plane is numerically uncertifiable"; return false; }
    solidtriangle::Interval squared{};
    for (float value:normal) squared=solidtriangle::add(squared,solidtriangle::mul(solidtriangle::exact(value),solidtriangle::exact(value)));
    if (!(squared.low>0)) { error="Box triangle response has no certified normal"; return false; }
    const double distance=solidtriangle::up((solidtriangle::up((reach+2*first.tolerance)*length)-separation)/squared.low);
    if (!std::isfinite(distance)) { error="Box triangle response distance is not finite"; return false; }
    candidate.x=static_cast<float>(double(candidate.x)+normal[0]*distance);
    candidate.y=static_cast<float>(double(candidate.y)+normal[1]*distance);
    candidate.z=static_cast<float>(double(candidate.z)+normal[2]*distance);
    if (!solidtriangle::normalCoordinates(solidPosition(candidate)))
    { error="Box triangle response requires finite normal-range coordinates"; return false; }
    for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
        if (triangle!=firstTriangle) {
            solidtriangle::Query projected; bool hit=false;
            if (!solidtriangle::boxSweep(responseFrom,solidPosition(candidate),box,vertices(triangle),reach,
                budget,projected,hit,error)) return false;
            if (hit) { error="Box triangle response projection may cross another surface; this frame is unsupported"; return false; }
        }
        solidtriangle::Query final;
        if (!solidtriangle::boxQuery(solidPosition(candidate),solidPosition(candidate),box,vertices(triangle),0,final,budget)
            || final.lower<reach-final.tolerance)
        { error=budget.used>=budget.limit ? "Triangle contact replay exceeds 100M checked work units":
            "Box triangle response cannot certify separation from every surface"; return false; }
    }
    const auto velocityBefore=solidVelocity(candidate);
    solidBoxSurfaceImpulse(candidate,box,arm,normal,{},p);
    if (impulse) {
        const auto velocityAfter=solidVelocity(candidate);
        for (int axis=0;axis<3;++axis) impulse->impulse[axis]=double(p.bodyMass)*(double(velocityAfter[axis])-velocityBefore[axis]);
        impulse->valid=std::any_of(impulse->impulse.begin(),impulse->impulse.end(),[](double value) { return value!=0; });
        if (impulse->valid) {
            impulse->bodyIdentity=candidate.identity; impulse->bodyGeometryIdentity=candidate.geometryIdentity;
            impulse->meshIdentity=frame.meshIdentity; impulse->triangle=firstTriangle;
            for (int corner=0;corner<3;++corner)
                impulse->vertexIdentities[corner]=frame.meshVertexIdentities[frame.meshTriangles[firstTriangle][corner]];
            impulse->weights=first.weights; impulse->point=first.closest; impulse->time=first.time;
        }
    }
    return true;
}

inline bool solidGeometryContact(ParticleRigidBody& body,const ParticleRigidBody& previous,
                                 const ParticleGeometryFrame& frame,const ParticleGeometryFrame& old,
                                 float dt,const ParticleParams& p,bool sweep=true,
                                 const SolidVector* boxHalf=nullptr,SolidTriangleWorkBudget* work=nullptr,std::string* diagnostic=nullptr,
                                 bool correctionSweep=false,SolidTriangleContactImpulse* contactImpulse=nullptr,
                                 std::uint64_t expectedMeshIdentity=0,const SolidRotatingSweep* rotating=nullptr)
{
    if (contactImpulse) *contactImpulse={};
    SolidTriangleContactImpulse stagedImpulse;
    const bool observe=contactImpulse && expectedMeshIdentity!=0 && frame.meshIdentity==expectedMeshIdentity;
    auto candidate=body;
    SolidTriangleWorkBudget localWork;
    auto& budget=work ? *work:localWork;
    std::string localError;
    auto& error=diagnostic ? *diagnostic:localError;
    if (frame.meshIdentity!=0 || old.meshIdentity!=0) {
        if (!std::isfinite(dt) || dt<=0 || !std::isfinite(p.bodyRadius) || p.bodyRadius<std::numeric_limits<float>::min()
            || !std::isfinite(p.colliderThickness) || p.colliderThickness<0
            || (p.colliderThickness!=0 && p.colliderThickness<std::numeric_limits<float>::min()))
        { error="Triangle surfaces require admitted bodies and a positive interval"; return false; }
        if (!validateSolidTriangleFrame(frame,budget,error) || !validateSolidTriangleFrame(old,budget,error)) return false;
        if (frame.meshIdentity!=old.meshIdentity || frame.meshVertexCount!=old.meshVertexCount
            || frame.meshTriangleCount!=old.meshTriangleCount
            || frame.meshVertexIdentities!=old.meshVertexIdentities || frame.meshTriangles!=old.meshTriangles)
        { error="Triangle surface changed its stable indexed topology during replay"; return false; }
        if (boxHalf) {
            if (!solidBoxTriangleContact(candidate,previous,frame,old,p,sweep,correctionSweep,*boxHalf,budget,error,
                observe ? &stagedImpulse:nullptr,dt,rotating)) return false;
        } else {
            const auto from=sweep || correctionSweep ? solidPosition(previous):solidPosition(candidate),to=solidPosition(candidate);
            if (!solidtriangle::normalCoordinates(from) || !solidtriangle::normalCoordinates(to))
            { error="Triangle contacts require finite normal-range sphere coordinates"; return false; }
            bool found=false,firstMoving=false,movingMesh=false; std::uint32_t firstTriangle=0; solidtriangle::Query first;
            const double reach=double(p.bodyRadius)+p.colliderThickness;
            for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
                std::array<SolidVector,3> before,current;
                bool moving=false;
                for (int corner=0;corner<3;++corner) {
                    const auto index=frame.meshTriangles[triangle][corner];
                    before[corner]=sweep ? old.meshVertices[index]:frame.meshVertices[index];
                    current[corner]=frame.meshVertices[index];
                    moving=moving || before[corner]!=current[corner];
                }
                if (moving) {
                    // Existing sampled endpoint geometry defines linear vertex motion
                    // across this exact replay interval, including event-split steps.
                    movingMesh=true;
                    if (!budget.charge()) { error="Triangle contact replay exceeds 100M checked work units"; return false; }
                    if (!solidtriangle::nondegenerateMotion(before,current))
                    { error="Moving triangle may degenerate during its retained linear interval; this frame is unsupported"; return false; }
                }
                solidtriangle::Query contact; bool hit=false;
                if (!solidtriangle::sweep(from,to,before,current,reach,budget,contact,hit,error)) return false;
                if (hit && (!found || contact.time<first.time))
                { found=true; first=contact; firstTriangle=triangle; firstMoving=moving; }
            }
            if (found) {
                SolidVector normal;
                for (int axis=0;axis<3;++axis) normal[axis]=static_cast<float>(first.normal[axis]);
                candidate.x=static_cast<float>(first.closest[0]+first.normal[0]*(reach+2*first.tolerance));
                candidate.y=static_cast<float>(first.closest[1]+first.normal[1]*(reach+2*first.tolerance));
                candidate.z=static_cast<float>(first.closest[2]+first.normal[2]*(reach+2*first.tolerance));
                const SolidVector responseFrom{static_cast<float>(first.position[0]),static_cast<float>(first.position[1]),static_cast<float>(first.position[2])};
                if (!solidtriangle::normalCoordinates(responseFrom) || !solidtriangle::normalCoordinates(solidPosition(candidate)))
                { error="Triangle response requires finite normal-range sphere coordinates"; return false; }
                // Projection happens at the retained impact time. Reuse the original
                // vertex endpoints, so no rounded intermediate mesh becomes authority.
                for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
                    std::array<SolidVector,3> before,vertices;
                    for (int corner=0;corner<3;++corner) {
                        const auto index=frame.meshTriangles[triangle][corner];
                        before[corner]=sweep ? old.meshVertices[index]:frame.meshVertices[index];
                        vertices[corner]=frame.meshVertices[index];
                    }
                    if (triangle!=firstTriangle) {
                        solidtriangle::Query projected; bool hit=false;
                        if (!solidtriangle::sweep(responseFrom,solidPosition(candidate),before,vertices,reach,budget,projected,hit,error,
                            first.time,first.time)) return false;
                        if (hit) { error="Triangle response projection may cross another surface; this frame is unsupported"; return false; }
                    }
                    if (!budget.charge()) { error="Triangle contact replay exceeds 100M checked work units"; return false; }
                    solidtriangle::Query final;
                    if (!solidtriangle::query(solidPosition(candidate),solidPosition(candidate),before,vertices,0,final,
                            solidtriangle::exact(first.time)) || final.lower<reach-final.tolerance)
                    { error="Triangle contact response cannot certify separation from every surface"; return false; }
                }
                SolidVector surface{};
                if (firstMoving) for (int axis=0;axis<3;++axis) {
                    double velocity=0;
                    for (int corner=0;corner<3;++corner) {
                        const auto index=frame.meshTriangles[firstTriangle][corner];
                        velocity+=first.weights[corner]*(double(frame.meshVertices[index][axis])-old.meshVertices[index][axis])/dt;
                    }
                    if (!std::isfinite(velocity)) { error="Moving triangle surface velocity is not finite"; return false; }
                    surface[axis]=static_cast<float>(std::clamp(velocity,-32.0,32.0));
                }
                SolidVector velocityBefore{};
                if (observe) velocityBefore=solidVelocity(candidate);
                solidImpulse(candidate,nullptr,normal,surface,p);
                if (observe) {
                    const auto velocityAfter=solidVelocity(candidate);
                    // Report the actual float velocity change after its component clamp.
                    for (int axis=0;axis<3;++axis)
                        stagedImpulse.impulse[axis]=double(p.bodyMass)*(double(velocityAfter[axis])-double(velocityBefore[axis]));
                    stagedImpulse.valid=std::any_of(stagedImpulse.impulse.begin(),stagedImpulse.impulse.end(),
                        [](double value) { return value!=0; });
                    if (stagedImpulse.valid) {
                        stagedImpulse.bodyIdentity=candidate.identity;
                        stagedImpulse.bodyGeometryIdentity=candidate.geometryIdentity;
                        stagedImpulse.meshIdentity=frame.meshIdentity;
                        stagedImpulse.triangle=firstTriangle;
                        for (int corner=0;corner<3;++corner)
                            stagedImpulse.vertexIdentities[corner]=frame.meshVertexIdentities[frame.meshTriangles[firstTriangle][corner]];
                        stagedImpulse.weights=first.weights;
                        stagedImpulse.point=first.closest;
                        stagedImpulse.time=first.time;
                    }
                }
                if (movingMesh) {
                    const auto remainingFrom=solidPosition(candidate),velocity=solidVelocity(candidate);
                    const double remaining=double(dt)*(1-first.time);
                    candidate.x=static_cast<float>(double(remainingFrom[0])+remaining*velocity[0]);
                    candidate.y=static_cast<float>(double(remainingFrom[1])+remaining*velocity[1]);
                    candidate.z=static_cast<float>(double(remainingFrom[2])+remaining*velocity[2]);
                    if (!solidtriangle::normalCoordinates(solidPosition(candidate)))
                    { error="Moving triangle response requires finite normal-range sphere coordinates"; return false; }
                    for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
                        std::array<SolidVector,3> before,vertices;
                        for (int corner=0;corner<3;++corner) {
                            const auto index=frame.meshTriangles[triangle][corner];
                            before[corner]=old.meshVertices[index]; vertices[corner]=frame.meshVertices[index];
                        }
                        solidtriangle::Query remainingContact; bool hit=false;
                        if (!solidtriangle::sweep(remainingFrom,solidPosition(candidate),before,vertices,reach,budget,
                            remainingContact,hit,error,first.time,1)) return false;
                        if (hit) { error="Moving triangle remaining response may cross a surface; this frame is unsupported"; return false; }
                        if (!budget.charge()) { error="Triangle contact replay exceeds 100M checked work units"; return false; }
                        solidtriangle::Query final;
                        if (!solidtriangle::query(solidPosition(candidate),solidPosition(candidate),vertices,vertices,0,final)
                            || final.lower<reach-final.tolerance)
                        { error="Moving triangle response cannot certify final separation from every surface"; return false; }
                    }
                }
            }
        }
    }
    const auto verifyCorrection=[&](const SolidVector& from) {
        if (!frame.meshIdentity || from==solidPosition(candidate)) return true;
        if (!solidtriangle::normalCoordinates(from) || !solidtriangle::normalCoordinates(solidPosition(candidate)))
        { error="Triangle correction requires finite normal-range sphere coordinates"; return false; }
        // Check each correction separately. A later correction cannot cancel
        // an earlier crossing and disguise it as a safe net displacement.
        for (std::uint32_t triangle=0;triangle<frame.meshTriangleCount;++triangle) {
            std::array<SolidVector,3> vertices;
            for (int corner=0;corner<3;++corner) vertices[corner]=frame.meshVertices[frame.meshTriangles[triangle][corner]];
            solidtriangle::Query corrected; bool hit=false;
            if (boxHalf) {
                if (!solidtriangle::boxSweep(from,solidPosition(candidate),solidBox(candidate,*boxHalf),vertices,
                    p.colliderThickness,budget,corrected,hit,error)) return false;
            } else if (!solidtriangle::sweep(from,solidPosition(candidate),vertices,vertices,
                double(p.bodyRadius)+p.colliderThickness,budget,corrected,hit,error)) return false;
            if (hit) { error="Point or boundary correction may cross a triangle surface; this frame is unsupported"; return false; }
        }
        return true;
    };
    for (int i=0;i<frame.pointCount;++i) {
        const auto beforeCorrection=solidPosition(candidate);
        SolidVector surface;
        for (int axis=0;axis<3;++axis) surface[axis]=std::clamp(
            (frame.solidPoints[i][axis]-old.solidPoints[i][axis])/dt,-32.0f,32.0f);
        if (boxHalf) {
            solidBoxSphereContact(candidate,*boxHalf,frame.solidPoints[i],p.colliderThickness,surface,p);
            if (!verifyCorrection(beforeCorrection)) return false;
            continue;
        }
        if (sweep) solidSphereSweep(candidate,nullptr,solidPosition(previous),old.solidPoints[i],
            frame.solidPoints[i],p.colliderThickness,surface,p);
        if (!verifyCorrection(beforeCorrection)) return false;
        const auto beforeDiscrete=solidPosition(candidate);
        solidSphereContact(candidate,nullptr,frame.solidPoints[i],p.colliderThickness,surface,p);
        if (!verifyCorrection(beforeDiscrete)) return false;
    }
    if (p.collisionMode!=0) for (int axis=0;axis<3;++axis) {
        const float value=solidPosition(candidate)[axis];
        if (value<p.bodyRadius || value>1-p.bodyRadius) {
            const auto beforeCorrection=solidPosition(candidate);
            SolidVector normal {}; normal[axis]=value<p.bodyRadius ? 1:-1;
            solidMove(candidate,normal,value<p.bodyRadius ? p.bodyRadius-value:value-(1-p.bodyRadius));
            solidImpulse(candidate,nullptr,normal,{},p);
            if (!verifyCorrection(beforeCorrection)) return false;
        }
    }
    body=candidate;
    if (contactImpulse && stagedImpulse.valid) *contactImpulse=stagedImpulse;
    return true;
}
}
