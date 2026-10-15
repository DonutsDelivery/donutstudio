#pragma once

#include "particle_body_geometry_admission.h"
#include "particle_history.h"
#include <mutex>

namespace videorender {

class ParticleBodyCallerLease;

struct ParticleBodyGeometrySampleRecipe final
{
    ParticleBodyGeometryWork derivation, admittedEvaluation, incrementalEvaluation;
    std::size_t derivationLogicalBytes=0, retainedBaseLogicalBytes=0;
    std::size_t retainedSampleLogicalBytes=0, attemptLogicalBytes=0;
    std::size_t derivationControlVisits=0, controlVisits=0, perAttemptVisits=0, totalVisits=0;
};

struct ParticleBodyGeometrySampleLimits final
{
    std::size_t maxDerivationVisits=100000;
    std::size_t maxRetainedLogicalBytes=videowire::geometry::kMaximumBufferBytes;
    // Includes the shared source, every escaped sample and concurrent attempts.
    std::size_t maxAttemptLogicalBytes=videowire::geometry::kMaximumBufferBytes;
    std::size_t maxTotalVisits=1000000000;
};

struct ParticleBodyGeometrySampleAttempt final
{
    ParticleBodyGeometrySampleRecipe recipe;
    bool derivationReserved=false, evaluationReserved=false, published=false;
};

struct ParticleBodyGeometrySampleUsage final
{
    std::size_t retainedLogicalBytes=0, reservedRetainedLogicalBytes=0, attemptLogicalBytes=0, attempts=0, visits=0;
};

namespace particlebodygeometrysampledetail {
struct Ledger final
{
    const std::shared_ptr<const ParticleBodyCallerLease> callerResources;
    const std::shared_ptr<const ParticleBodyGeometryAdmission> admission;
    const ParticleBodyGeometrySampleRecipe recipe;
    const ParticleBodyGeometrySampleLimits limits;
    std::mutex mutex;
    ParticleBodyGeometrySampleUsage usage;
    Ledger(std::shared_ptr<const ParticleBodyGeometryAdmission> a,ParticleBodyGeometrySampleRecipe r,
        ParticleBodyGeometrySampleLimits l,std::shared_ptr<const ParticleBodyCallerLease> resources)
        : callerResources(std::move(resources)),admission(std::move(a)),recipe(r),limits(l)
    { usage.retainedLogicalBytes=r.retainedBaseLogicalBytes; }
};
struct Lease final
{
    const std::shared_ptr<Ledger> ledger;
    const std::size_t bytes;
    bool active=false;
    Lease(std::shared_ptr<Ledger> l,std::size_t b) : ledger(std::move(l)),bytes(b) {}
    ~Lease()
    {
        if (active) { std::lock_guard<std::mutex> lock(ledger->mutex); ledger->usage.retainedLogicalBytes-=bytes; }
    }
};
struct Reservation final
{
    const std::shared_ptr<Ledger> ledger;
    std::size_t bytes=0, retainedBytes=0;
    explicit Reservation(std::shared_ptr<Ledger> l) : ledger(std::move(l)) {}
    ~Reservation()
    {
        if (bytes) {
            std::lock_guard<std::mutex> lock(ledger->mutex);
            ledger->usage.attemptLogicalBytes-=bytes;
            ledger->usage.reservedRetainedLogicalBytes-=retainedBytes;
        }
    }
    bool reserve()
    {
        using namespace videowire::geometry;
        std::lock_guard<std::mutex> lock(ledger->mutex);
        auto& u=ledger->usage; const auto& r=ledger->recipe; const auto& l=ledger->limits;
        std::size_t retained=0,pending=0,attempt=0,total=0,visits=0;
        if (u.attempts>=ledger->admission->recipe().replayBoundaries
            || !checkedAdd(u.reservedRetainedLogicalBytes,r.retainedSampleLogicalBytes,pending)
            || !checkedAdd(u.retainedLogicalBytes,pending,retained)
            || retained>l.maxRetainedLogicalBytes
            || !checkedAdd(u.attemptLogicalBytes,r.attemptLogicalBytes,attempt)
            || !checkedAdd(u.retainedLogicalBytes,attempt,total) || total>l.maxAttemptLogicalBytes
            || !checkedAdd(u.visits,r.perAttemptVisits,visits) || visits>l.maxTotalVisits) return false;
        u.attemptLogicalBytes=attempt; u.reservedRetainedLogicalBytes=pending; u.visits=visits; ++u.attempts;
        bytes=r.attemptLogicalBytes; retainedBytes=r.retainedSampleLogicalBytes;
        return true;
    }
    void retain(Lease& lease)
    {
        std::lock_guard<std::mutex> lock(ledger->mutex);
        ledger->usage.attemptLogicalBytes-=lease.bytes; bytes-=lease.bytes;
        ledger->usage.reservedRetainedLogicalBytes-=retainedBytes; retainedBytes=0;
        ledger->usage.retainedLogicalBytes+=lease.bytes; lease.active=true;
    }
};
} // namespace particlebodygeometrysampledetail

class ParticleBodyGeometrySampler;
inline bool deriveParticleBodyGeometrySampleRecipe(
    const std::shared_ptr<const ParticleBodyGeometryAdmission>&,const ParticleBodyGeometrySampleLimits&,
    ParticleBodyGeometrySampleAttempt&,std::size_t&,std::string&);

class ParticleBodyGeometrySample final
{
public:
    ParticleBodyGeometrySample(const ParticleBodyGeometrySample&)=delete;
    ParticleBodyGeometrySample(ParticleBodyGeometrySample&&)=delete;
    ParticleBodyGeometrySample& operator=(const ParticleBodyGeometrySample&)=delete;
    ParticleBodyGeometrySample& operator=(ParticleBodyGeometrySample&&)=delete;
    const std::shared_ptr<const ParticleBodyGeometryAdmission>& admission() const noexcept { return data_->lease->ledger->admission; }
    const std::shared_ptr<const videowire::geometry::AdmittedValue>& source() const noexcept { return admission()->source(); }
    const videowire::geometry::BudgetReceipt& sourceReceipt() const noexcept { return admission()->sourceReceipt(); }
    std::int64_t absoluteTick() const noexcept { return data_->tick; }
    static constexpr std::int64_t ticksPerSecond=kParticleReplayTicksPerSecond;
    std::uint64_t structuralRevision() const noexcept { return admission()->structuralRevision(); }
    const videowire::geometry::GeometryData& prototype() const noexcept { return data_->normalized; }
    const videowire::geometry::Vec3& center() const noexcept { return data_->center; }
    float radius() const noexcept { return data_->radius; }
    const std::array<float,3>& boxHalfRatios() const noexcept { return data_->half; }
private:
    struct Data final
    {
        const std::shared_ptr<const particlebodygeometrysampledetail::Lease> lease;
        const videowire::geometry::GeometryData normalized;
        const videowire::geometry::Vec3 center;
        const float radius;
        const std::array<float,3> half;
        const std::int64_t tick;
        Data(std::shared_ptr<const particlebodygeometrysampledetail::Lease> l,
            videowire::geometry::GeometryData mesh,videowire::geometry::Vec3 c,float r,
            std::array<float,3> h,std::int64_t t)
            : lease(std::move(l)),normalized(std::move(mesh)),center(c),radius(r),half(h),tick(t) {}
    };
    explicit ParticleBodyGeometrySample(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    const std::shared_ptr<const Data> data_;
    friend class ParticleBodyGeometrySampler;
    friend bool deriveParticleBodyGeometrySampleRecipe(
        const std::shared_ptr<const ParticleBodyGeometryAdmission>&,const ParticleBodyGeometrySampleLimits&,
        ParticleBodyGeometrySampleAttempt&,std::size_t&,std::string&);
    friend std::shared_ptr<const ParticleBodyGeometrySampler> makeParticleBodyGeometrySampler(
        const std::shared_ptr<const ParticleBodyGeometryAdmission>&,const ParticleBodyGeometrySampleLimits&,
        ParticleBodyGeometrySampleAttempt&,std::string&,const std::shared_ptr<const ParticleBodyCallerLease>&);
};

inline std::shared_ptr<const ParticleBodyGeometrySampler> makeParticleBodyGeometrySampler(
    const std::shared_ptr<const ParticleBodyGeometryAdmission>&,const ParticleBodyGeometrySampleLimits&,
    ParticleBodyGeometrySampleAttempt&,std::string&,const std::shared_ptr<const ParticleBodyCallerLease>& = {});

class ParticleBodyGeometrySampler final
{
public:
    ParticleBodyGeometrySampler(const ParticleBodyGeometrySampler&)=delete;
    ParticleBodyGeometrySampler& operator=(const ParticleBodyGeometrySampler&)=delete;
    const std::shared_ptr<const ParticleBodyGeometryAdmission>& admission() const noexcept { return ledger_->admission; }
    const ParticleBodyGeometrySampleRecipe& recipe() const noexcept { return ledger_->recipe; }
    ParticleBodyGeometrySampleUsage usage() const
    { std::lock_guard<std::mutex> lock(ledger_->mutex); return ledger_->usage; }

    std::shared_ptr<const ParticleBodyGeometrySample> sample(
        const std::shared_ptr<const videowire::geometry::AdmittedValue>& exactSource,
        std::uint64_t expectedSourceRevision,std::int64_t tick,
        ParticleBodyGeometrySampleAttempt& attempt,std::string& error) const
    {
        using namespace videowire::geometry;
        using namespace particlebodygeometrysampledetail;
        attempt={}; attempt.recipe=recipe(); error.clear();
        const auto refuse=[&](const char* reason)->std::shared_ptr<const ParticleBodyGeometrySample> { error=reason; return {}; };
        const auto& body=*ledger_->admission;
        // Exact integer identity is retained. The released field authority uses
        // double seconds and may round nearby rational times to equal outputs.
        // Reject integers outside double's consecutive exact-integer range.
        if (tick<0 || tick>9007199254740991LL)
            return refuse("Body geometry sample requires a representable nonnegative absolute tick");
        if (exactSource!=body.source() || expectedSourceRevision!=body.structuralRevision())
            return refuse("Body geometry sample requires its exact admitted token and source revision");
        Reservation reservation(ledger_);
        if (!reservation.reserve()) return refuse("Body geometry sample prospective storage or work reservation is short");
        attempt.evaluationReserved=true;
        std::size_t comparison=0;
        if (!body.recipe().sourceComparison.total(comparison) || !body.matchesExactSource(exactSource,comparison))
            return refuse("Body geometry sample source differs from its frozen admitted snapshot");
        try {
            ValueDescriptor working=body.ownedDescriptor(),replayed;
            MaterializedInstancePlan materialized;
            RuntimeFieldEvaluation runtime;
            runtime.timelineSeconds=double(tick)/kParticleReplayTicksPerSecond;
            if (!std::isfinite(runtime.timelineSeconds)
                || !validateOperationPlan(working,{},&materialized,error,&replayed,&runtime)
                || !particleGeometryIntermediatesFit(materialized,error)
                || !validateParticleGeometry(replayed,CarrierKind::geometry3D,error,false)) return {};
            const auto* mesh=std::get_if<GeometryData>(&replayed.data);
            const auto& reference=body.referencePrototype();
            if (!mesh || mesh->positions.size()!=reference.positions.size()
                || mesh->vertexIds!=reference.vertexIds || mesh->indices!=reference.indices)
                return refuse("Body geometry sample changed its admitted topology");
            for (const auto& v:mesh->positions)
                if (!finite(v)) return refuse("Body geometry sample contains a nonfinite position");
            for (auto index:mesh->indices)
                if (index>=mesh->positions.size()) return refuse("Body geometry sample index is out of range");
            auto low=mesh->positions.front(),high=low;
            for (const auto& v:mesh->positions) {
                low.x=std::min(low.x,v.x); low.y=std::min(low.y,v.y); low.z=std::min(low.z,v.z);
                high.x=std::max(high.x,v.x); high.y=std::max(high.y,v.y); high.z=std::max(high.z,v.z);
            }
            const Vec3 center{(low.x+high.x)*0.5f,(low.y+high.y)*0.5f,(low.z+high.z)*0.5f};
            float radius=0;
            for (const auto& v:mesh->positions)
                radius=std::max(radius,std::hypot(std::hypot(v.x-center.x,v.y-center.y),v.z-center.z));
            if (!finite(center) || !finite(radius) || radius<=1.0e-7f)
                return refuse("Body geometry sample has no finite enclosing radius");
            const std::array<float,3> half{(high.x-low.x)*0.5f/radius,(high.y-low.y)*0.5f/radius,(high.z-low.z)*0.5f/radius};
            if (!std::all_of(half.begin(),half.end(),[](float v) { return finite(v); }))
                return refuse("Body geometry sample has nonfinite normalized half ratios");
            // Authored bodyRadius and the released positive contact half-extent
            // threshold are downstream checks, not appearance mass or inertia.
            if (body.proxy()==ParticleBodyGeometryProxy::fixedBox && half!=body.referenceBoxHalfRatios())
                return refuse("Body geometry fixed-box sample changed its reference half ratios");
            GeometryData normalized;
            normalized.positions.reserve(mesh->positions.size());
            normalized.vertexIds=mesh->vertexIds; normalized.indices=mesh->indices;
            for (const auto& v:mesh->positions) {
                const Vec3 position{(v.x-center.x)/radius,(v.y-center.y)/radius,(v.z-center.z)/radius};
                if (!finite(position)) return refuse("Body geometry sample normalization is nonfinite");
                normalized.positions.push_back(position);
            }
            auto lease=std::make_shared<Lease>(ledger_,recipe().retainedSampleLogicalBytes);
            reservation.retain(*lease);
            // Copy into the by-value producer argument deliberately. The admitted
            // three-sample envelope does not depend on NRVO or copy elision.
            auto data=std::make_shared<const ParticleBodyGeometrySample::Data>(lease,normalized,center,radius,half,tick);
            auto result=std::shared_ptr<const ParticleBodyGeometrySample>(new ParticleBodyGeometrySample(std::move(data)));
            attempt.published=true;
            return result;
        } catch (const std::bad_alloc&) {
            return refuse("Body geometry sample allocation failed before publication");
        }
    }
private:
    explicit ParticleBodyGeometrySampler(std::shared_ptr<particlebodygeometrysampledetail::Ledger> ledger)
        : ledger_(std::move(ledger)) {}
    const std::shared_ptr<particlebodygeometrysampledetail::Ledger> ledger_;
    friend bool deriveParticleBodyGeometrySampleRecipe(
        const std::shared_ptr<const ParticleBodyGeometryAdmission>&,const ParticleBodyGeometrySampleLimits&,
        ParticleBodyGeometrySampleAttempt&,std::size_t&,std::string&);
    friend std::shared_ptr<const ParticleBodyGeometrySampler> makeParticleBodyGeometrySampler(
        const std::shared_ptr<const ParticleBodyGeometryAdmission>&,const ParticleBodyGeometrySampleLimits&,
        ParticleBodyGeometrySampleAttempt&,std::string&,const std::shared_ptr<const ParticleBodyCallerLease>&);
};

inline bool deriveParticleBodyGeometrySampleRecipe(
    const std::shared_ptr<const ParticleBodyGeometryAdmission>& body,const ParticleBodyGeometrySampleLimits& limits,
    ParticleBodyGeometrySampleAttempt& attempt,std::size_t& derivationVisits,std::string& error)
{
    using namespace videowire::geometry;
    using namespace particlebodygeometrysampledetail;
    attempt={}; derivationVisits=0; error.clear();
    const auto refuse=[&](const char* reason)->bool { error=reason; return false; };
    if (!body) return refuse("Body geometry sampler requires an immutable source admission");
    auto& r=attempt.recipe;
    r.derivation=body->recipe().census;
    r.derivationLogicalBytes=particlebodygeometrydetail::censusLogicalBytes;
    // total() visits nine named counters. Factory calls it three times; each
    // sample calls it once here and once inside matchesExactSource(). These are
    // explicit source loops, separate from geometry/field element visits.
    r.derivationControlVisits=27; r.controlVisits=18;
    if (!r.derivation.total(derivationVisits)
        || !checkedAdd(derivationVisits,r.derivationControlVisits,derivationVisits)
        || derivationVisits>limits.maxDerivationVisits
        || r.derivationLogicalBytes>limits.maxAttemptLogicalBytes)
        return refuse("Body geometry sampler derivation reservation is short");
    attempt.derivationReserved=true;
    r.admittedEvaluation=body->recipe().replayEvaluation;
    // Audit the accepted prospective envelope against validFieldElement's four
    // component visits. Its typed-component counts omit inactive zero checks.
    // Retained Mesh, TRS and Position Field each also call zeroParameters(),
    // visiting four parameters beyond the common finite/default-settings checks.
    // Retained Mesh and Position Field also build logical unique-ID sets
    // absent from that recipe.
    std::array<particlebodygeometrydetail::Shape,32> shapes{};
    std::size_t count=0,maxIdentityVertices=0;
    const auto find=[&](StableId id)->const particlebodygeometrydetail::Shape* {
        for (std::size_t i=0;i<count;++i) if (shapes[i].id==id) return &shapes[i];
        return nullptr;
    };
    const auto add=[](std::size_t& target,std::size_t value) { return checkedAdd(target,value,target); };
    for (const auto& op:body->ownedDescriptor().operations) {
        particlebodygeometrydetail::Shape out; out.id=op.outputStableId;
        const auto* a=find(op.inputStableId); const auto* b=find(op.secondaryInputStableId);
        std::size_t correction=0;
        switch (op.code) {
        case OperationCode::retainedMesh:
            if (!op.retainedMesh) return refuse("Body geometry sampler retained source is missing");
            out.vertices=op.retainedMesh->geometry.positions.size(); out.indices=op.retainedMesh->geometry.indices.size(); correction=4;
            maxIdentityVertices=std::max(maxIdentityVertices,out.vertices);
            if (!add(r.incrementalEvaluation.identityElements,out.vertices)) return refuse("Body geometry sampler work overflow");
            break;
        case OperationCode::grid:
            out.vertices=std::size_t(op.parameters[0])*std::size_t(op.parameters[1]);
            out.indices=(std::size_t(op.parameters[0])-1)*(std::size_t(op.parameters[1])-1)*6; break;
        case OperationCode::transformTRS:
            if (!a) return refuse("Body geometry sampler transform shape is missing");
            out.vertices=a->vertices; out.indices=a->indices; correction=4; break;
        case OperationCode::positionField:
            if (!a) return refuse("Body geometry sampler position shape is missing");
            out.fields=a->vertices;
            if (!checkedAdd(out.fields,4,correction)) return refuse("Body geometry sampler work overflow");
            maxIdentityVertices=std::max(maxIdentityVertices,out.fields);
            if (!add(r.incrementalEvaluation.identityElements,out.fields)) return refuse("Body geometry sampler work overflow");
            break;
        case OperationCode::fieldVectorMeasure:
        case OperationCode::fieldTimeline:
        case OperationCode::fieldComposeVector:
            if (!a) return refuse("Body geometry sampler field shape is missing");
            out.fields=a->fields;
            if (!checkedMul(out.fields,op.code==OperationCode::fieldTimeline ? 6:4,correction))
                return refuse("Body geometry sampler work overflow");
            break;
        case OperationCode::vectorDisplacement:
            if (!a || !b) return refuse("Body geometry sampler displacement shape is missing");
            out.vertices=a->vertices; out.indices=a->indices; correction=out.vertices; break;
        default: return refuse("Body geometry sampler operation has no source-work schedule");
        }
        if (count>=shapes.size() || !add(r.incrementalEvaluation.fieldComponents,correction))
            return refuse("Body geometry sampler work overflow");
        shapes[count++]=out;
    }
    // Reserve all the new explicit loops separately. This intentionally keeps
    // the old prospective normalization/copies too, rather than claiming credits
    // for overlap. One final finite pass and index range pass are included.
    const auto& mesh=body->referencePrototype();
    if (!checkedMul(mesh.positions.size(),4,r.incrementalEvaluation.positionElements)
        || !add(r.incrementalEvaluation.identityElements,mesh.vertexIds.size())
        || !checkedMul(mesh.indices.size(),2,r.incrementalEvaluation.indexElements)
        || !add(r.incrementalEvaluation.fieldComponents,3)) return refuse("Body geometry sampler work overflow");
    std::size_t baseline=0,increment=0;
    if (!r.admittedEvaluation.total(baseline) || !r.incrementalEvaluation.total(increment)
        || !checkedAdd(baseline,increment,r.perAttemptVisits)
        || !checkedAdd(r.perAttemptVisits,r.controlVisits,r.perAttemptVisits)
        || !checkedMul(r.perAttemptVisits,body->recipe().replayBoundaries,r.totalVisits)
        || !checkedAdd(r.totalVisits,derivationVisits,r.totalVisits) || r.totalVisits>limits.maxTotalVisits)
        return refuse("Body geometry sampler total work reservation is short or overflowed");
    r.retainedBaseLogicalBytes=body->recipe().retainedLogicalBytes;
    r.retainedSampleLogicalBytes=body->recipe().sampleLogicalBytes;
    // Logical objects and size-derived arrays only. Allocation/control-block
    // headers, spare STL capacities, tree balancing, libm and driver internals
    // remain unknown. This is not an allocator or elapsed-time bound.
    if (!add(r.retainedBaseLogicalBytes,sizeof(Ledger)+sizeof(ParticleBodyGeometrySampler))
        || !add(r.retainedSampleLogicalBytes,sizeof(ParticleBodyGeometrySample::Data)-sizeof(GeometryData)
            +sizeof(ParticleBodyGeometrySample)+sizeof(Lease))) return refuse("Body geometry sampler storage overflow");
    // The admission's prospective attempt already includes its shared retained
    // source/reference. The ledger charges that closure once, outside each
    // concurrent attempt, so remove that exact component here.
    if (body->recipe().replayAttemptLogicalBytes<body->recipe().retainedLogicalBytes)
        return refuse("Body geometry sampler prospective storage is inconsistent");
    r.attemptLogicalBytes=body->recipe().replayAttemptLogicalBytes-body->recipe().retainedLogicalBytes;
    if (!addArrayBytes(3,r.retainedSampleLogicalBytes-body->recipe().sampleLogicalBytes,r.attemptLogicalBytes)
        || !addArrayBytes(maxIdentityVertices,sizeof(StableId),r.attemptLogicalBytes)
        || !add(r.attemptLogicalBytes,2*sizeof(std::set<StableId>)+sizeof(std::set<std::string>)
            +sizeof(RuntimeFieldEvaluation)+sizeof(Reservation)))
        return refuse("Body geometry sampler storage overflow");
    std::size_t retained=0,peak=0,creation=0;
    if (!checkedAdd(r.retainedBaseLogicalBytes,r.retainedSampleLogicalBytes,retained)
        || retained>limits.maxRetainedLogicalBytes
        || !checkedAdd(r.retainedBaseLogicalBytes,r.attemptLogicalBytes,peak) || peak>limits.maxAttemptLogicalBytes
        || !checkedAdd(r.retainedBaseLogicalBytes,r.derivationLogicalBytes,creation) || creation>limits.maxAttemptLogicalBytes)
        return refuse("Body geometry sampler prospective storage reservation is short");
    return true;
}

inline std::shared_ptr<const ParticleBodyGeometrySampler> makeParticleBodyGeometrySampler(
    const std::shared_ptr<const ParticleBodyGeometryAdmission>& body,const ParticleBodyGeometrySampleLimits& limits,
    ParticleBodyGeometrySampleAttempt& attempt,std::string& error,
    const std::shared_ptr<const ParticleBodyCallerLease>& callerResources)
{
    using namespace particlebodygeometrysampledetail;
    std::size_t derivationVisits=0;
    if (!deriveParticleBodyGeometrySampleRecipe(body,limits,attempt,derivationVisits,error)) return {};
    const auto& r=attempt.recipe;
    const auto refuse=[&](const char* reason)->std::shared_ptr<const ParticleBodyGeometrySampler> { error=reason; return {}; };
    try {
        auto ledger=std::make_shared<Ledger>(body,r,limits,callerResources);
        ledger->usage.visits=derivationVisits;
        return std::shared_ptr<const ParticleBodyGeometrySampler>(new ParticleBodyGeometrySampler(std::move(ledger)));
    } catch (const std::bad_alloc&) {
        return refuse("Body geometry sampler allocation failed before publication");
    }
}
} // namespace videorender
