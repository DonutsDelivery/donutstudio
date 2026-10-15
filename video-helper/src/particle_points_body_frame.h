#pragma once

#include "particle_body_replay.h"
#include "../../shared/GeometryCoreScene.h"
#include <string_view>

namespace videorender { class ParticlePointsBodySourceAdmission; }
namespace videowire {
inline std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission> admitParticlePointsBodySource(
    const std::string&,const std::string&,const geometry::DeferredBodyGeometrySource&,
    const std::shared_ptr<videorender::ParticleBodyCallerLedger>&,std::string&);
}
namespace videorender {

inline constexpr std::size_t particlePointsBodyIssuanceCensusWork=32*sizeof(videowire::geometry::OperationRecord)
    +sizeof(videowire::geometry::DeferredBodyGeometrySource);
inline constexpr std::size_t particlePointsBodyCoreOwnerVisits=32;
inline constexpr std::size_t particlePointsBodyNativeHeaderVisits=64;
struct ParticlePointsBodyEnvelope final {
    std::size_t bytes=0, work=0;
};

class ParticlePointsBodySourceAdmission final {
public:
    ParticlePointsBodySourceAdmission(const ParticlePointsBodySourceAdmission&)=delete;
    ParticlePointsBodySourceAdmission& operator=(const ParticlePointsBodySourceAdmission&)=delete;
    const videowire::geometry::DeferredBodyGeometrySource& declaration() const noexcept { return declaration_; }
    const ParticleParams& parameters() const noexcept { return parameters_; }
    const std::string& providerPayload() const noexcept { return payload_; }
    const std::string& bodyPlan() const noexcept { return bodyPlan_; }
    const std::shared_ptr<const ParticleBodyGeometryAdmission>& body() const noexcept { return body_; }
    const std::shared_ptr<ParticleBodyCallerLedger>& caller() const noexcept { return lease_->ledger; }

    // Shape-only preflight. No descriptor/mesh comparison, decode or array copy
    // runs before this envelope has been reserved on the persistent caller.
    static bool issuanceEnvelope(const videowire::geometry::DeferredBodyGeometrySource& source,
        const std::string& payload,const std::string& bodyPlan,ParticlePointsBodyEnvelope& envelope)
    {
        using namespace videowire::geometry;
        if (!source.source || !source.reference || payload.size()>131072 || bodyPlan.size()>65536) return false;
        const auto& value=source.source->descriptor();
        const auto* mesh=std::get_if<GeometryData>(&value.data);
        if (!mesh || mesh->positions.size()>256 || mesh->indices.size()>384 || mesh->vertexIds.size()>256
            || value.operations.empty() || value.operations.size()>32 || !value.attributes.empty()
            || !value.spectrumFields.empty() || !value.scoreFields.empty()
            || source.reference->positions.size()>256 || source.reference->vertexIds.size()>256
            || source.reference->indices.size()>384) return false;
        std::size_t copies=sizeof(ValueDescriptor), capacities=sizeof(AdmittedValue);
        if (!addArrayBytes(mesh->positions.size(),sizeof(Vec3),copies)
            || !addArrayBytes(mesh->vertexIds.size(),sizeof(StableId),copies)
            || !addArrayBytes(mesh->indices.size(),sizeof(std::uint32_t),copies)
            || !addArrayBytes(value.operations.size(),sizeof(OperationRecord),copies)
            || !addArrayBytes(mesh->positions.capacity(),sizeof(Vec3),capacities)
            || !addArrayBytes(mesh->vertexIds.capacity(),sizeof(StableId),capacities)
            || !addArrayBytes(mesh->indices.capacity(),sizeof(std::uint32_t),capacities)
            || !addArrayBytes(value.operations.capacity(),sizeof(OperationRecord),capacities)) return false;
        for (const auto& op:value.operations) {
            if (op.sdfSurface || !op.field.attributeName.empty()) return false;
            if (!checkedAdd(capacities,op.field.attributeName.capacity()+1,capacities)) return false;
            if (op.retainedMesh) {
                const auto& retained=*op.retainedMesh;
                if (!retained.attributes.empty() || !retained.importedAnimation.empty()
                    || retained.geometry.positions.size()>256 || retained.geometry.vertexIds.size()>256
                    || retained.geometry.indices.size()>384) return false;
                if (!checkedAdd(copies,sizeof(RetainedMeshData),copies)
                    || !addArrayBytes(retained.geometry.positions.size(),sizeof(Vec3),copies)
                    || !addArrayBytes(retained.geometry.vertexIds.size(),sizeof(StableId),copies)
                    || !addArrayBytes(retained.geometry.indices.size(),sizeof(std::uint32_t),copies)
                    || !checkedAdd(capacities,sizeof(RetainedMeshData),capacities)
                    || !addArrayBytes(retained.geometry.positions.capacity(),sizeof(Vec3),capacities)
                    || !addArrayBytes(retained.geometry.vertexIds.capacity(),sizeof(StableId),capacities)
                    || !addArrayBytes(retained.geometry.indices.capacity(),sizeof(std::uint32_t),capacities)
                    || !checkedAdd(capacities,retained.importedAnimation.capacity()+1,capacities)) return false;
            }
        }
        // Eight descriptor owners cover decoder/admit copies, its compatibility
        // descriptor, the body factory snapshot/reference and concurrent scratch.
        // The supported 32-record body schedule visits at most 32*32 shapes and
        // 32 bounded 256-vertex/384-index mesh or vertex-field intermediates per
        // pass. Decode, validation, comparison and publication have eight passes.
        envelope.bytes=sizeof(ParticlePointsBodySourceAdmission)+sizeof(ParticleParams)
            +sizeof(ParticleGeometryBinding)+particlebodygeometrydetail::censusLogicalBytes;
        envelope.work=8u*32u*(32u+256u*16u+384u*4u);
        if (source.reference.get()!=mesh
            && (!addArrayBytes(source.reference->positions.capacity(),sizeof(Vec3),envelope.bytes)
                || !addArrayBytes(source.reference->vertexIds.capacity(),sizeof(StableId),envelope.bytes)
                || !addArrayBytes(source.reference->indices.capacity(),sizeof(std::uint32_t),envelope.bytes))) return false;
        std::size_t descriptors=0, strings=0;
        return checkedMul(copies,8,descriptors) && checkedAdd(envelope.bytes,descriptors,envelope.bytes)
            && checkedAdd(envelope.bytes,capacities,envelope.bytes)
            && checkedAdd(payload.capacity(),bodyPlan.capacity(),strings)
            && checkedAdd(strings,2,strings) && checkedMul(strings,8,strings)
            && checkedAdd(envelope.bytes,strings,envelope.bytes)
            && checkedAdd(envelope.work,envelope.bytes,envelope.work);
    }
private:
    // Reservation precedes every owner so its refund follows their destruction.
    const std::shared_ptr<const ParticleBodyCallerLease> lease_;
    const ParticleParams parameters_;
    const std::string payload_,bodyPlan_;
    const std::shared_ptr<const ParticleBodyGeometryAdmission> body_;
    const videowire::geometry::DeferredBodyGeometrySource declaration_;
    ParticlePointsBodySourceAdmission(std::shared_ptr<const ParticleBodyCallerLease> lease,
        ParticleParams parameters,std::string payload,std::string bodyPlan,
        std::shared_ptr<const ParticleBodyGeometryAdmission> body,
        videowire::geometry::DeferredBodyGeometrySource declaration)
        : lease_(std::move(lease)),parameters_(std::move(parameters)),payload_(std::move(payload)),
          bodyPlan_(std::move(bodyPlan)),body_(std::move(body)),declaration_(std::move(declaration)) {}
    friend std::shared_ptr<const ParticlePointsBodySourceAdmission> videowire::admitParticlePointsBodySource(
        const std::string&,const std::string&,const videowire::geometry::DeferredBodyGeometrySource&,
        const std::shared_ptr<videorender::ParticleBodyCallerLedger>&,std::string&);
};

class ParticlePointsBodyRequest final {
public:
    ParticlePointsBodyRequest(const ParticlePointsBodyRequest&)=delete;
    ParticlePointsBodyRequest& operator=(const ParticlePointsBodyRequest&)=delete;
    const std::shared_ptr<const ParticlePointsBodySourceAdmission>& source() const noexcept { return source_; }
    const ParticleParams& parameters() const noexcept { return parameters_; }
    double clipSeconds() const noexcept { return clipSeconds_; }
    float aspect() const noexcept { return float(parameters_.bodyReplayContext->width)/parameters_.bodyReplayContext->height; }
    const canonicalblockc::CanonicalBlockCFrame* frame() const noexcept { return parameters_.bodyReplayContext->frame.get(); }
    const videohelper::geometry::PlanOwnerIdentity& owner() const noexcept { return parameters_.bodyReplayContext->owner; }

    static std::shared_ptr<const ParticlePointsBodyRequest> create(
        const std::shared_ptr<const ParticlePointsBodySourceAdmission>& source,
        const std::shared_ptr<ParticleBodyCallerLedger>& caller,
        const videohelper::geometry::PlanOwnerIdentity& owner,int nodeId,int width,int height,
        const std::shared_ptr<const ParticleHistorySource>& history,
        const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
        double clipSeconds,double projectSeconds,std::string& error)
    {
        error.clear();
        if (!source || caller!=source->caller() || !caller || !caller->onCallerThread()
            || !owner.valid() || owner.planGeneration!=source->declaration().structuralRevision
            || nodeId<=0 || static_cast<videowire::geometry::StableId>(nodeId)+1!=source->declaration().pointsOutputStableId
            || owner.clipIdentity>static_cast<std::uint64_t>(std::numeric_limits<int>::max())
            || width<=0 || height<=0 || !history || !history->score
            || !std::isfinite(clipSeconds) || !std::isfinite(projectSeconds)) {
            error="Paired particle request requires its admitted source and current caller, owner, history and clocks"; return {};
        }
        auto lease=caller->reserve(sizeof(ParticlePointsBodyRequest)+2*sizeof(ParticleParams),2*sizeof(ParticleParams));
        if (!lease) { error="Paired particle request prospective admission is short"; return {}; }
        auto context=makeParticleBodyReplayContext(caller,error);
        if (!context) return {};
        context->owner=owner; context->width=width; context->height=height; context->frame=frame;
        auto parameters=source->parameters();
        parameters.history=history; parameters.historyProjectSeconds=projectSeconds;
        parameters.historyClipId=static_cast<int>(owner.clipIdentity); parameters.historyNodeId=nodeId;
        parameters.geometryRevision=owner.planGeneration; parameters.bodyReplayContext=std::move(context);
        if (owner.projectGeneration!=history->key.projectGeneration
            || owner.helperGeneration!=history->key.helperGeneration || owner.deviceGeneration!=history->key.deviceGeneration
            || (frame && (!(frame->key==history->key) || !canonicalblockc::valid(frame)))
            || (!history->score->notes.empty() && (!frame
                || history->key.beat!=static_cast<double>(static_cast<float>(history->timeline.secondsToBeat(projectSeconds)))))
            || !particleHistoryReady(parameters,error,clipSeconds)) {
            if (error.empty()) error="Paired particle request differs from the exact current history/frame owner";
            return {};
        }
        try { return std::shared_ptr<const ParticlePointsBodyRequest>(new ParticlePointsBodyRequest(
            std::move(lease),source,std::move(parameters),clipSeconds)); }
        catch (const std::bad_alloc&) { error="Paired particle request allocation failed"; return {}; }
    }
private:
    const std::shared_ptr<const ParticleBodyCallerLease> lease_;
    const std::shared_ptr<const ParticlePointsBodySourceAdmission> source_;
    const ParticleParams parameters_;
    const double clipSeconds_;
    ParticlePointsBodyRequest(std::shared_ptr<const ParticleBodyCallerLease> lease,
        std::shared_ptr<const ParticlePointsBodySourceAdmission> source,ParticleParams parameters,double seconds)
        : lease_(std::move(lease)),source_(std::move(source)),parameters_(std::move(parameters)),clipSeconds_(seconds) {}
};

class ParticlePointsBodyFrame final {
public:
    ParticlePointsBodyFrame(const ParticlePointsBodyFrame&)=delete;
    ParticlePointsBodyFrame& operator=(const ParticlePointsBodyFrame&)=delete;
    const std::shared_ptr<const ParticlePointsBodyRequest>& request() const noexcept { return request_; }
    const std::shared_ptr<const ParticleBodyReplayReceipt>& receipt() const noexcept { return receipt_; }
    const videowire::geometry::ValueDescriptor& points() const noexcept { return points_; }
    const std::shared_ptr<const ParticleBodyGeometrySample>& sample() const noexcept { return sample_; }
    videowire::geometry::BodyGeometryFrameData geometry() const noexcept {
        return {std::shared_ptr<const videowire::geometry::GeometryData>(sample_,&sample_->prototype()),
            sample_->source(),request_->source()->declaration().structuralRevision,sample_->absoluteTick()};
    }
    bool matches(const ParticlePointsBodyRequest& current,std::string& error) const {
        const auto& p=current.parameters();
        auto lease=current.source()->caller()->reserve(sizeof(ParticleParams),sizeof(ParticleParams));
        if (!lease) { error="Paired particle completed-frame comparison admission is short"; return false; }
        auto prepared=p;
        prepared.preparedBodyReplay=receipt_; prepared.preparedBodies=receipt_->bodies(); prepared.preparedSolids=receipt_->solids();
        if (current.source()!=request_->source() || current.clipSeconds()!=request_->clipSeconds()
            || !receipt_->matchesPrepared(prepared,current.clipSeconds(),current.aspect(),current.frame())
            || !receipt_->matchesRender(prepared,current.owner(),p.historyNodeId,p.bodyReplayContext->width,p.bodyReplayContext->height)
            || !sample_ || sample_!=receipt_->solids()->terminalBodySample || !receipt_->sampler()
            || receipt_->sampler()->admission()!=sample_->admission()
            || sample_->source()!=current.source()->declaration().source
            || sample_->admission()->source()!=p.geometryBinding->bodySource
            || sample_->absoluteTick()!=receipt_->terminalTick()
            || sample_->structuralRevision()!=current.source()->declaration().sourceRevision
            || points_.stableId!=current.source()->declaration().pointsOutputStableId
            || points_.sourceStableId!=current.source()->declaration().points.sourceStableId
            || points_.sourceRevision!=current.source()->declaration().points.sourceRevision) {
            error="Paired particle completed frame does not match the independent current request"; return false;
        }
        return true;
    }

    // Metadata from the admitted source only. No replay, geometry evaluation,
    // sampler allocation or completed-state authority is produced here.
    static bool replayEnvelope(const ParticlePointsBodyRequest& request,
        ParticlePointsBodyEnvelope& conversion,std::size_t& callerPeak,std::string& error)
    {
        using namespace videowire::geometry;
        const auto& input=request.parameters();
        const auto& binding=request.source()->declaration().points;
        if (!hasDynamicParticleBody(input) || input.motionMode!=2 || !input.historicalReplay || input.simulationSpace!=1
            || input.count<1 || input.count>kParticleSolidMaxBodies || binding.capacity!=static_cast<std::size_t>(input.count)) {
            error="Paired particle replay requires the admitted dynamic Solid 3D Coupled body route"; return false;
        }
        conversion.bytes=sizeof(ParticlePointsBodyFrame)+sizeof(ParticlePointReplayTimes)+sizeof(ValueDescriptor)
            +2*sizeof(ParticleParams)+sizeof(ParticleSolidState);
        if (!addArrayBytes(binding.capacity,sizeof(PointRecord)+3*sizeof(FieldElement)+sizeof(StableId)+4*sizeof(void*),conversion.bytes)
            || !addArrayBytes(3,sizeof(AttributeData)+kMaximumNameBytes+1,conversion.bytes)
            || !checkedMul(binding.capacity,64+32,conversion.work)
            || !checkedAdd(conversion.work,conversion.bytes,conversion.work)) {
            error="Paired particle conversion resource arithmetic overflow"; return false;
        }
        const auto& admitted=request.source()->body();
        const auto& body=admitted->recipe();
        std::size_t derivation=0;
        if (!body.census.total(derivation) || !checkedAdd(derivation,27,derivation)
            || !request.source()->caller()->charge(derivation)) {
            error="Paired particle sampler recipe derivation work is short"; return false;
        }
        ParticleBodyGeometrySampleAttempt sample;
        std::size_t visits=0;
        if (!deriveParticleBodyGeometrySampleRecipe(admitted,input.bodyReplayContext->sampleLimits,sample,visits,error)) return false;
        // Storage recipes are independent of replay boundary count. The fresh
        // factory shares the already leased AdmittedValue and source arrays;
        // its new frozen descriptor/reference and sampler ledger are additional.
        std::size_t shared=0,base=0,factory=0,sampling=0,creation=0,geometry=0;
        if (!checkedAdd(sizeof(AdmittedValue),body.sourceLogicalBytes,shared)
            || body.factoryAttemptLogicalBytes<shared || sample.recipe.retainedBaseLogicalBytes<shared) {
            error="Paired particle shared-source resource recipe is inconsistent"; return false;
        }
        factory=body.factoryAttemptLogicalBytes-shared;
        base=sample.recipe.retainedBaseLogicalBytes-shared;
        // sampleHistory retains its old bodySample while the candidate sample
        // evaluates. The attempt already includes its three publication copies.
        if (!checkedAdd(base,sample.recipe.attemptLogicalBytes,sampling)
            || !checkedAdd(sampling,sample.recipe.retainedSampleLogicalBytes,sampling)
            || !checkedAdd(base,sample.recipe.derivationLogicalBytes,creation)) {
            error="Paired particle geometry overlap arithmetic overflow"; return false;
        }
        geometry=std::max(factory,std::max(creation,sampling));
        std::size_t additional=geometry;
        if (!checkedAdd(additional,sizeof(ParticleBodyGeometryAttempt)+sizeof(ParticleBodyGeometrySampleAttempt)+sizeof(ParticlePointsBodyEnvelope)
                +sizeof(GeometryResourcesOwner)+3*sizeof(std::shared_ptr<const ParticleBodyCallerLease>),additional)
            || !checkedAdd(conversion.bytes,additional,conversion.bytes)
            || !checkedAdd(conversion.work,additional,conversion.work)) {
            error="Paired particle geometry caller resource arithmetic overflow"; return false;
        }
        // Completed-frame matches keeps one prepared Params lease while each
        // of its two sequential receipt comparisons holds a validation lease.
        std::size_t validation=ParticleBodyReplayReceipt::retainedLogicalBytes()
            +sizeof(ParticleParams)+ParticleBodyReplayReceipt::validationLogicalBytes()+3*sizeof(ParticleBodyCallerLease);
        callerPeak=std::max(ParticleBodyReplayState::callerFreshPeakLogicalBytes(true,true),validation);
        if (!checkedAdd(callerPeak,conversion.bytes,callerPeak)
            || !checkedAdd(callerPeak,sizeof(ParticleBodyCallerLease),callerPeak)) {
            error="Paired particle caller peak resource arithmetic overflow"; return false;
        }
        return true;
    }

    static std::shared_ptr<const ParticlePointsBodyFrame> replay(
        const std::shared_ptr<const ParticlePointsBodyRequest>& request,std::string& error)
    {
        using namespace videowire::geometry;
        error.clear();
        try {
        if (!request) { error="Paired particle replay has no admitted current request"; return {}; }
        const auto& input=request->parameters();
        const auto& binding=request->source()->declaration().points;
        if (!hasDynamicParticleBody(input) || input.motionMode!=2 || !input.historicalReplay || input.simulationSpace!=1
            || input.count<1 || input.count>kParticleSolidMaxBodies || binding.capacity!=static_cast<std::size_t>(input.count)) {
            error="Paired particle replay requires the admitted dynamic Solid 3D Coupled body route"; return {};
        }
        ParticlePointsBodyEnvelope conversion;
        std::size_t callerPeak=0;
        if (!replayEnvelope(*request,conversion,callerPeak,error)) return {};
        // This caller lease covers the new geometry factory/sample closure once.
        // The independent geometry ledgers continue to admit the same objects.
        auto lease=request->source()->caller()->reserve(conversion.bytes,conversion.work);
        if (!lease) { error="Paired particle conversion prospective admission is short"; return {}; }
        ParticleSolidState state;
        ParticlePointReplayTimes times;
        std::shared_ptr<const ParticleBodyReplayReceipt> receipt;
        // Keep the already leased shared source closure alive for sample,
        // sampler and copied-admission escapes without charging it again.
        auto resourcesOwner=std::make_shared<const GeometryResourcesOwner>(GeometryResourcesOwner{lease,request->source()});
        const auto geometryResources=std::shared_ptr<const ParticleBodyCallerLease>(resourcesOwner,lease.get());
        replayParticleBodies(input,request->clipSeconds(),request->aspect(),request->frame(),&error,&state,&times,&receipt,geometryResources);
        if (!error.empty() || !receipt || !receipt->solids() || !receipt->solids()->terminalBodySample) {
            if (error.empty()) error="Paired particle replay did not issue a genuine terminal receipt and sample";
            return {};
        }
        ValueDescriptor value;
        value.carrier=CarrierKind::points3D; value.stableId=request->source()->declaration().pointsOutputStableId;
        value.sourceStableId=binding.sourceStableId; value.sourceRevision=binding.sourceRevision; value.data=PointsData{};
        value.attributes={{{1,"velocity",ValueType::vector,Domain::point,Interpolation::nearest},{}},
                          {{2,"age",ValueType::floatValue,Domain::point,Interpolation::nearest},{}},
                          {{3,"lifetime",ValueType::floatValue,Domain::point,Interpolation::nearest},{}}};
        auto currentParameters=input;
        if (!sampleParticleParameters(input,currentParameters,times.sampleSeconds,error)) return {};
        auto& points=std::get<PointsData>(value.data).points;
        points.reserve(binding.capacity);
        for (auto& attribute:value.attributes) attribute.elements.reserve(binding.capacity);
        std::set<StableId> identities;
        const auto& solids=*receipt->solids();
        for (int i=0;i<solids.count;++i) {
            const auto& body=solids.bodies[static_cast<std::size_t>(i)];
            if (body.geometryIdentity==0 && body.noteRow<0) {
                error="Particle Points3D emitters require explicit stable point identities"; return {};
            }
            const auto id=body.geometryIdentity!=0 ? body.geometryIdentity : body.identity>0
                ? static_cast<StableId>(body.identity)*2 : static_cast<StableId>(-(body.identity+1))*2+1;
            if (!validStableId(id) || !identities.insert(id).second) {
                error="Particle identities must be unique and representable as Points3D"; return {};
            }
            points.push_back({{body.x,body.y,body.z},id,currentParameters.bodyRadius,
                {body.orientation[0],body.orientation[1],body.orientation[2],body.orientation[3]}});
            value.attributes[0].elements.push_back({{body.vx,body.vy,body.vz,0}});
            value.attributes[1].elements.push_back({{std::max(0.0,times.sampleSeconds-times.birthSeconds[i]),0,0,0}});
            value.attributes[2].elements.push_back({{times.lifetimeSeconds[i],0,0,0}});
        }
        auto result=std::shared_ptr<const ParticlePointsBodyFrame>(new ParticlePointsBodyFrame(
            std::move(lease),request,std::move(receipt),times,std::move(value)));
        return result->matches(*request,error) ? result:nullptr;
        } catch (const std::bad_alloc&) { error="Paired particle replay allocation failed before publication"; return {}; }
    }
private:
    struct GeometryResourcesOwner final {
        const std::shared_ptr<const ParticleBodyCallerLease> lease;
        const std::shared_ptr<const ParticlePointsBodySourceAdmission> source;
    };
    const std::shared_ptr<const ParticleBodyCallerLease> lease_;
    const std::shared_ptr<const ParticlePointsBodyRequest> request_;
    const std::shared_ptr<const ParticleBodyReplayReceipt> receipt_;
    const std::shared_ptr<const ParticleBodyGeometrySample> sample_;
    const ParticlePointReplayTimes times_;
    const videowire::geometry::ValueDescriptor points_;
    ParticlePointsBodyFrame(std::shared_ptr<const ParticleBodyCallerLease> lease,
        std::shared_ptr<const ParticlePointsBodyRequest> request,
        std::shared_ptr<const ParticleBodyReplayReceipt> receipt,ParticlePointReplayTimes times,
        videowire::geometry::ValueDescriptor points)
        : lease_(std::move(lease)),request_(std::move(request)),receipt_(std::move(receipt)),
          sample_(receipt_->solids()->terminalBodySample),times_(times),points_(std::move(points)) {}
};
} // namespace videorender

namespace videohelper::geometry {
// Resource recipe only; no source, request or completed-state authority.
bool particlePointsBodyCoreEnvelope(const videowire::geometry::DeferredBodyGeometrySource&,std::size_t,
    bool,videorender::ParticlePointsBodyEnvelope&);
}
