#pragma once

#include "../../shared/ParticleGeometryContract.h"
#include <new>

namespace videorender {

class ParticleBodyCallerLease;

enum class ParticleBodyGeometryProxy { sphere, fixedBox };

// Bounds source-level loop visits, not allocator, STL tree, libm or driver work.
// Previously executed GCRV decoding/admitValue work is outside this new factory.
struct ParticleBodyGeometryWork final
{
    std::size_t operationRecords=0, shapeProbes=0, dependencyEdges=0;
    std::size_t positionElements=0, identityElements=0, indexElements=0;
    std::size_t fieldElements=0, fieldComponents=0, stringBytes=0;

    bool total(std::size_t& result) const noexcept
    {
        result=0;
        for (auto count : {operationRecords,shapeProbes,dependencyEdges,positionElements,
                identityElements,indexElements,fieldElements,fieldComponents,stringBytes})
            if (!videowire::geometry::checkedAdd(result,count,result)) return false;
        return true;
    }
};

struct ParticleBodyGeometryRecipe final
{
    // Geometry evaluation/normalization/owner transfers only. Historical inputs,
    // contacts, scenes, native resources and compositor leases need their own
    // admission before a future replay or draw may consume this prerequisite.
    std::size_t replayBoundaries=0;
    std::size_t retainedLogicalBytes=0, factoryAttemptLogicalBytes=0;
    std::size_t replayAttemptLogicalBytes=0, replayTotalVisits=0;
    // The shared source and reference are counted once, independently of body count.
    std::size_t sourceLogicalBytes=0, descriptorCopyLogicalBytes=0;
    std::size_t retainedSourceSnapshotLogicalBytes=0;
    std::size_t materializedLogicalBytes=0, sampleLogicalBytes=0;
    std::size_t censusLogicalBytes=0;
    ParticleBodyGeometryWork census, referenceValidation, referenceNormalization;
    ParticleBodyGeometryWork publication, sourceComparison, replayEvaluation;
};

struct ParticleBodyGeometryLimits final
{
    std::size_t maxReplayBoundaries=100000;
    std::size_t maxRetainedLogicalBytes=videowire::geometry::kMaximumBufferBytes;
    std::size_t maxFactoryAttemptLogicalBytes=videowire::geometry::kMaximumBufferBytes;
    std::size_t maxReplayAttemptLogicalBytes=videowire::geometry::kMaximumBufferBytes;
    std::size_t maxCensusVisits=100000, maxFactoryVisits=1000000;
    std::size_t maxReplayVisits=1000000000;
};

// Failure preserves this receipt and the caller's original source. A reservation
// flag records admission to a phase; it is not proof that the phase completed.
struct ParticleBodyGeometryAttempt final
{
    ParticleBodyGeometryRecipe recipe;
    bool censusReserved=false, referenceReserved=false, publicationReserved=false;
    bool published=false;
};

// Borrowed wire facts and observed capacities are cost inputs, never admission
// authority. A serialized issuer validates its scalar schedule before using them.
struct ParticleBodyGeometryScalarOperation final
{
    videowire::geometry::OperationCode code{};
    videowire::geometry::StableId operationId=0,inputId=0,outputId=0,secondaryId=0,auxiliaryId=0;
    std::array<float,4> parameters{};
    std::array<float,10> transform{};
    videowire::geometry::Domain fieldDomain{};
    videowire::geometry::ValueType fieldType{};
    std::uint8_t fieldMode=0;
    std::array<double,8> fieldValues{};
    std::size_t vertices=0,indices=0,fields=0;
    std::size_t retainedVertices=0,retainedIndices=0;
    std::size_t inputIndex=32,secondaryIndex=32,auxiliaryIndex=32;
    bool emptyPayloads=false;
};
struct ParticleBodyGeometryScalarCensus final
{
    std::array<ParticleBodyGeometryScalarOperation,32> operations{};
    std::size_t operationCount=0,terminalVertices=0,terminalIndices=0;
    videowire::geometry::StableId terminalId=0;
};
struct ParticleBodyGeometryOwnedCapacityRecord final
{
    std::size_t attributeName=0;
    bool retained=false;
    std::size_t positions=0,identities=0,indices=0,attributes=0,importedAnimation=0;
};
struct ParticleBodyGeometryOwnedCapacityCensus final
{
    const videowire::geometry::AdmittedValue* observedSource=nullptr;
    std::size_t operationCount=0,terminalVertices=0,terminalIndices=0;
    std::size_t positions=0,identities=0,indices=0,operations=0,attributes=0,spectrumFields=0,scoreFields=0;
    std::array<ParticleBodyGeometryOwnedCapacityRecord,32> records{};
};
// Direct reset stores name native field slots. Padding is not explicitly stored.
// A later observer reservation pays this reset separately from C0 construction.
namespace particlebodycapacityreset {
struct StoreSite final { const char* name; std::size_t nativeBytes; };
inline constexpr std::array<StoreSite,11> globalStores{{
    {"observedSource",sizeof(ParticleBodyGeometryOwnedCapacityCensus::observedSource)},
    {"operationCount",sizeof(ParticleBodyGeometryOwnedCapacityCensus::operationCount)},
    {"terminalVertices",sizeof(ParticleBodyGeometryOwnedCapacityCensus::terminalVertices)},
    {"terminalIndices",sizeof(ParticleBodyGeometryOwnedCapacityCensus::terminalIndices)},
    {"positions",sizeof(ParticleBodyGeometryOwnedCapacityCensus::positions)},
    {"identities",sizeof(ParticleBodyGeometryOwnedCapacityCensus::identities)},
    {"indices",sizeof(ParticleBodyGeometryOwnedCapacityCensus::indices)},
    {"operations",sizeof(ParticleBodyGeometryOwnedCapacityCensus::operations)},
    {"attributes",sizeof(ParticleBodyGeometryOwnedCapacityCensus::attributes)},
    {"spectrumFields",sizeof(ParticleBodyGeometryOwnedCapacityCensus::spectrumFields)},
    {"scoreFields",sizeof(ParticleBodyGeometryOwnedCapacityCensus::scoreFields)}
}};
inline constexpr std::array<StoreSite,7> recordStores{{
    {"attributeName",sizeof(ParticleBodyGeometryOwnedCapacityRecord::attributeName)},
    {"retained",sizeof(ParticleBodyGeometryOwnedCapacityRecord::retained)},
    {"positions",sizeof(ParticleBodyGeometryOwnedCapacityRecord::positions)},
    {"identities",sizeof(ParticleBodyGeometryOwnedCapacityRecord::identities)},
    {"indices",sizeof(ParticleBodyGeometryOwnedCapacityRecord::indices)},
    {"attributes",sizeof(ParticleBodyGeometryOwnedCapacityRecord::attributes)},
    {"importedAnimation",sizeof(ParticleBodyGeometryOwnedCapacityRecord::importedAnimation)}
}};
inline constexpr std::size_t recordLoopVisits=
    std::tuple_size<decltype(ParticleBodyGeometryOwnedCapacityCensus::records)>::value;
inline constexpr std::size_t scalarStores=globalStores.size()+recordStores.size()*recordLoopVisits;
inline constexpr std::size_t directNativeByteSlots=[]() constexpr {
    std::size_t bytes=0;
    for (const auto& site:globalStores) bytes+=site.nativeBytes;
    for (const auto& site:recordStores) bytes+=site.nativeBytes*recordLoopVisits;
    return bytes;
}();
} // namespace particlebodycapacityreset

enum class ParticleBodyGeometryCostRefusal { none, unsupported, arithmetic, sourceMismatch };
struct ParticleBodyGeometryScalarCost final
{
    std::size_t operationCount=0,retainedCount=0,terminalVertices=0,terminalIndices=0;
    // H+A+R+Xterminal, then +Mterminal+C. Source S is measured later.
    std::size_t retainedConstant=0,factoryAttemptConstant=0;
    std::size_t retainedSourceCoefficient=1,factorySourceCoefficient=2;
    std::size_t nativeDataBytes=0,nativeBodyBytes=0,censusLogicalBytes=0;
    std::size_t descriptorCopyLogicalBytes=0,retainedSourceSnapshotLogicalBytes=0,sampleLogicalBytes=0;
    std::size_t factoryDependencyLogicalBytes=0,retainedHolderLogicalBytes=0;
    std::size_t sourceLogicalBytes=0,retainedLogicalBytes=0,factoryAttemptLogicalBytes=0;
    std::size_t owningFactoryAttemptLogicalBytes=0;
    ParticleBodyGeometryWork census,referenceValidation,referenceNormalization,publication;
};

class ParticleBodyGeometryAdmission;
inline std::shared_ptr<const ParticleBodyGeometryAdmission> admitParticleBodyGeometry(
    const std::shared_ptr<const videowire::geometry::AdmittedValue>&,
    ParticleBodyGeometryProxy, std::size_t, const ParticleBodyGeometryLimits&,
    ParticleBodyGeometryAttempt&, std::string&,const std::shared_ptr<const ParticleBodyCallerLease>& = {});
inline std::shared_ptr<const ParticleBodyGeometryAdmission> admitParticleBodyGeometryWithRetainedResources(
    const std::shared_ptr<const videowire::geometry::AdmittedValue>&,
    ParticleBodyGeometryProxy, std::size_t, const ParticleBodyGeometryLimits&,
    ParticleBodyGeometryAttempt&, std::string&,const std::shared_ptr<const ParticleBodyCallerLease>&,
    const videowire::geometry::RetainedResourceOwner&);

class ParticleBodyGeometryAdmission final
{
public:
    ParticleBodyGeometryAdmission(const ParticleBodyGeometryAdmission&)=default;
    ParticleBodyGeometryAdmission(ParticleBodyGeometryAdmission&&)=default;
    ParticleBodyGeometryAdmission& operator=(const ParticleBodyGeometryAdmission&)=delete;
    ParticleBodyGeometryAdmission& operator=(ParticleBodyGeometryAdmission&&)=delete;

    const std::shared_ptr<const videowire::geometry::AdmittedValue>& source() const noexcept { return data_->source; }
    const videowire::geometry::BudgetReceipt& sourceReceipt() const noexcept { return data_->receipt; }
    const videowire::geometry::ValueDescriptor& ownedDescriptor() const noexcept { return data_->snapshot; }
    const videowire::geometry::GeometryData& referencePrototype() const noexcept { return data_->normalized; }
    const videowire::geometry::Vec3& referenceCenter() const noexcept { return data_->center; }
    float referenceRadius() const noexcept { return data_->radius; }
    // Contact admission must still apply authored bodyRadius and the released
    // positive half-extent threshold. This reference admits no contact or inertia.
    const std::array<float,3>& referenceBoxHalfRatios() const noexcept { return data_->halfRatios; }
    ParticleBodyGeometryProxy proxy() const noexcept { return data_->proxy; }
    const ParticleBodyGeometryRecipe& recipe() const noexcept { return data_->recipe; }
    videowire::geometry::StableId terminalIdentity() const noexcept { return data_->snapshot.stableId; }
    videowire::geometry::StableId sourceIdentity() const noexcept { return data_->snapshot.sourceStableId; }
    std::uint64_t structuralRevision() const noexcept { return data_->snapshot.sourceRevision; }

    static bool deriveScalarCost(const ParticleBodyGeometryScalarCensus&,std::size_t,
        ParticleBodyGeometryProxy,ParticleBodyGeometryScalarCost&,ParticleBodyGeometryCostRefusal&) noexcept;
    static bool observeOwnedCapacities(const videowire::geometry::AdmittedValue&,
        ParticleBodyGeometryOwnedCapacityCensus&,ParticleBodyGeometryCostRefusal&) noexcept;
    static bool instantiateScalarCost(const videowire::geometry::AdmittedValue&,
        const ParticleBodyGeometryScalarCost&,const ParticleBodyGeometryOwnedCapacityCensus&,
        ParticleBodyGeometryScalarCost&,ParticleBodyGeometryCostRefusal&) noexcept;
    static videowire::geometry::RetainedResourceOwner makeSnapshotResources(
        const std::shared_ptr<const videowire::geometry::AdmittedValue>& source,
        const std::shared_ptr<const ParticleBodyCallerLease>& resources)
    {
        if (!source || !resources) return {};
        return std::make_shared<const ParticleBodyGeometryFactoryResources>(resources,source);
    }

    // This comparison checks its own prospective source-work allowance. A later sampler must
    // evaluate ownedDescriptor() through the existing authority, and bind the
    // resulting sample to absolute replay ticks. This predicate admits no mesh.
    bool matchesExactSource(const std::shared_ptr<const videowire::geometry::AdmittedValue>& candidate,
        std::size_t maxComparisonVisits) const noexcept
    {
        using namespace videowire::geometry;
        if (candidate!=data_->source) return false;
        std::size_t visits=0;
        if (!data_->recipe.sourceComparison.total(visits) || visits>maxComparisonVisits) return false;
        const auto& a=candidate->descriptor(); const auto& b=data_->snapshot;
        if (a.schemaVersion!=b.schemaVersion || a.carrier!=b.carrier || a.stableId!=b.stableId
            || a.sourceStableId!=b.sourceStableId || a.sourceRevision!=b.sourceRevision
            || a.overflow!=b.overflow || a.dispatchCount!=b.dispatchCount
            || a.fieldDomain!=b.fieldDomain || a.fieldValueType!=b.fieldValueType
            || a.fieldInterpolation!=b.fieldInterpolation || a.fieldDomainCardinality!=b.fieldDomainCardinality
            || a.operations.size()!=b.operations.size() || !a.attributes.empty()
            || !a.spectrumFields.empty() || !a.scoreFields.empty()
            || !std::holds_alternative<GeometryData>(a.data)
            || !equal(std::get<GeometryData>(a.data),std::get<GeometryData>(b.data))) return false;
        for (std::size_t i=0;i<a.operations.size();++i) {
            const auto& x=a.operations[i]; const auto& y=b.operations[i];
            if (x.stableId!=y.stableId || x.code!=y.code || x.inputStableId!=y.inputStableId
                || x.outputStableId!=y.outputStableId || x.secondaryInputStableId!=y.secondaryInputStableId
                || x.parameters!=y.parameters || !equal(x.transform,y.transform) || !(x.field==y.field)
                || x.sdfSurface || x.retainedMesh!=data_->retainedOwners[i]
                || bool(x.retainedMesh)!=bool(y.retainedMesh)) return false;
            if (x.retainedMesh && (!equal(x.retainedMesh->geometry,y.retainedMesh->geometry)
                || !x.retainedMesh->attributes.empty() || !x.retainedMesh->importedAnimation.empty())) return false;
        }
        return true;
    }

private:
    struct ParticleBodyGeometryFactoryResources final
    {
        const std::shared_ptr<const ParticleBodyCallerLease> lease;
        const std::shared_ptr<const videowire::geometry::AdmittedValue> source;
        ParticleBodyGeometryFactoryResources(std::shared_ptr<const ParticleBodyCallerLease> resources,
            std::shared_ptr<const videowire::geometry::AdmittedValue> token)
            : lease(std::move(resources)),source(std::move(token)) {}
    };
    struct Data final
    {
        const std::shared_ptr<const ParticleBodyCallerLease> callerResources;
        const std::shared_ptr<const videowire::geometry::AdmittedValue> source;
        const videowire::geometry::BudgetReceipt receipt;
        const videowire::geometry::ValueDescriptor snapshot;
        const videowire::geometry::GeometryData normalized;
        const std::array<std::shared_ptr<const videowire::geometry::RetainedMeshData>,32> retainedOwners;
        const videowire::geometry::Vec3 center;
        const float radius;
        const std::array<float,3> halfRatios;
        const ParticleBodyGeometryProxy proxy;
        const ParticleBodyGeometryRecipe recipe;
        Data(const std::shared_ptr<const videowire::geometry::AdmittedValue>& sourceIn,
            videowire::geometry::ValueDescriptor&& snapshotIn, videowire::geometry::GeometryData&& normalizedIn,
            std::array<std::shared_ptr<const videowire::geometry::RetainedMeshData>,32>&& owners,
            const videowire::geometry::Vec3& centerIn,const float& radiusIn,const std::array<float,3>& halfIn,
            const ParticleBodyGeometryProxy& proxyIn,const ParticleBodyGeometryRecipe& recipeIn,
            const std::shared_ptr<const ParticleBodyCallerLease>& resources)
            : callerResources(std::move(resources)),source(std::move(sourceIn)),receipt(source->receipt()),snapshot(std::move(snapshotIn)),
              normalized(std::move(normalizedIn)),retainedOwners(std::move(owners)),center(centerIn),
              radius(radiusIn),halfRatios(halfIn),proxy(proxyIn),recipe(recipeIn) {}
    };
    explicit ParticleBodyGeometryAdmission(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    const std::shared_ptr<const Data> data_;
    friend std::shared_ptr<const ParticleBodyGeometryAdmission> admitParticleBodyGeometry(
        const std::shared_ptr<const videowire::geometry::AdmittedValue>&,ParticleBodyGeometryProxy,
        std::size_t,const ParticleBodyGeometryLimits&,ParticleBodyGeometryAttempt&,std::string&,const std::shared_ptr<const ParticleBodyCallerLease>&);
    friend std::shared_ptr<const ParticleBodyGeometryAdmission> admitParticleBodyGeometryWithRetainedResources(
        const std::shared_ptr<const videowire::geometry::AdmittedValue>&,ParticleBodyGeometryProxy,
        std::size_t,const ParticleBodyGeometryLimits&,ParticleBodyGeometryAttempt&,std::string&,
        const std::shared_ptr<const ParticleBodyCallerLease>&,const videowire::geometry::RetainedResourceOwner&);
};

namespace particlebodygeometrydetail {
using namespace videowire::geometry;
struct Shape final { StableId id=0; std::size_t vertices=0,indices=0,fields=0; };
inline constexpr std::size_t censusLogicalBytes=sizeof(std::array<Shape,32>);

// These types are used only in template arguments and static assertions. They
// create no proof object, runtime calculator instruction or storage reservation.
template <std::size_t... Terms> struct NativeBytes;
template <> struct NativeBytes<> { static constexpr std::size_t value=0; };
template <std::size_t First,std::size_t... Rest> struct NativeBytes<First,Rest...> {
    static_assert(First<=std::numeric_limits<std::size_t>::max()-NativeBytes<Rest...>::value,
        "Body native event sum overflows");
    static constexpr std::size_t value=First+NativeBytes<Rest...>::value;
};
template <std::size_t Limit,std::size_t... Terms> struct NativeFits {
    static constexpr bool value=NativeBytes<Terms...>::value<=Limit;
};
template <std::size_t Count,std::size_t Width> struct NativeProduct {
    static_assert(!Count || Width<=std::numeric_limits<std::size_t>::max()/Count,
        "Body native event product overflows");
    static constexpr std::size_t value=Count*Width;
};

struct PositiveReferenceHalf final {
    bool operator()(float v) const { return finite(v) && v>0; }
};

// SC=V+X+n*(Op+1)+R+Q, A=X+n*(Op+1), Sample=G+X, with
// nonnegative capacity excess Q. The unchanged full reservation has fixed
// RF=B+Data+Admitted+2*V+G+C and coefficients 3/5/3/2 for n*u/X/R/Q.
// Terminal mesh ancestry is Grid or retainedMesh; descendants preserve counts.
// Grid gives X>=4*(Vec3+StableId)+6*uint32. After the radius guard retained
// ancestry gives X>=2*(Vec3+StableId)+3*uint32 and R>=RetainedMeshData+X+1.
// A one-vertex secondary retained root with Grid terminal uses R>=0 here.
// Both checked positive-sum comparisons retain the minimum payload credit.
// Coefficient rejection selects the primary template before any subtraction
// or residual multiplication can instantiate. These are type gates, not a
// numerical native PASS or runtime proof object.
template <std::size_t ReleasedFixed,std::size_t EventNative,std::size_t Source,
    std::size_t DescriptorArrays,std::size_t Retained,std::size_t Normalized,
    bool CoefficientsValid=NativeBytes<Source,DescriptorArrays>::value<=3
        && NativeBytes<Source,DescriptorArrays,Normalized>::value<=5
        && NativeBytes<Source,Retained>::value<=3 && Source<=2>
struct NativePayloadFits {
    static_assert(NativeBytes<Source,DescriptorArrays>::value<=3,
        "Body event operation/name coefficient exceeds reservation");
    static_assert(NativeBytes<Source,DescriptorArrays,Normalized>::value<=5,
        "Body event terminal-array coefficient exceeds reservation");
    static_assert(NativeBytes<Source,Retained>::value<=3,
        "Body event retained-array coefficient exceeds reservation");
    static_assert(Source<=2,"Body event source-capacity coefficient exceeds reservation");
    static constexpr bool grid=false,retained=false;
};
template <std::size_t ReleasedFixed,std::size_t EventNative,std::size_t Source,
    std::size_t DescriptorArrays,std::size_t Retained,std::size_t Normalized>
struct NativePayloadFits<ReleasedFixed,EventNative,Source,DescriptorArrays,Retained,Normalized,true> {
    using GridMinimum=NativeBytes<NativeProduct<4,NativeBytes<sizeof(Vec3),sizeof(StableId)>::value>::value,
        NativeProduct<6,sizeof(std::uint32_t)>::value>;
    using RetainedMinimum=NativeBytes<NativeProduct<2,NativeBytes<sizeof(Vec3),sizeof(StableId)>::value>::value,
        NativeProduct<3,sizeof(std::uint32_t)>::value>;
    using RetainedRootMinimum=NativeBytes<sizeof(RetainedMeshData),RetainedMinimum::value,1>;
    using OperationCredit=NativeProduct<3-NativeBytes<Source,DescriptorArrays>::value,
        NativeBytes<sizeof(OperationRecord),1>::value>;
    using Event=NativeBytes<EventNative,NativeProduct<Source,sizeof(ValueDescriptor)>::value>;
    static constexpr bool grid=NativeFits<NativeBytes<ReleasedFixed,OperationCredit::value,
        NativeProduct<5-NativeBytes<Source,DescriptorArrays,Normalized>::value,GridMinimum::value>::value>::value,
        Event::value>::value;
    static constexpr bool retained=NativeFits<NativeBytes<ReleasedFixed,OperationCredit::value,
        NativeProduct<5-NativeBytes<Source,DescriptorArrays,Normalized>::value,RetainedMinimum::value>::value,
        NativeProduct<3-NativeBytes<Source,Retained>::value,RetainedRootMinimum::value>::value>::value,
        Event::value>::value;
};

inline bool meshBytes(std::size_t vertices,std::size_t indices,std::size_t& bytes) noexcept
{
    bytes=sizeof(GeometryData);
    return addArrayBytes(vertices,sizeof(Vec3)+sizeof(StableId),bytes)
        && addArrayBytes(indices,sizeof(std::uint32_t),bytes);
}

// This census reads shape metadata only. It never evaluates source operations or
// walks mesh/field elements. Empty names, attributes and external bindings are an
// explicit first schedule, not an estimate of their general execution costs.
// Remaining schedules include other mesh generators, joining/bounds, normal and
// attribute operations, materials, curves/points/instances, field math/constants,
// searches/raycast, Score/audio/history, water, SDF and imported animation.
template <std::size_t OuterNativeBytes,std::size_t FactoryNativeFloor>
inline bool derive(const ValueDescriptor& value,std::size_t boundaries,ParticleBodyGeometryProxy proxy,
    ParticleBodyGeometryRecipe& r,std::string& error)
{
    std::array<Shape,32> shapes{};
    std::size_t shapeCount=0, geometryCount=0,fieldCount=0,edges=0;
    std::size_t meshPayload=0,fieldPayload=0,sourcePayload=0,temporaryMesh=0,temporaryField=0;
    std::size_t displacementCount=0,rootCount=0,constructedCount=0;
    const auto add=[](std::size_t& target,std::size_t count) { return checkedAdd(target,count,target); };
    const auto find=[&](StableId id)->const Shape* {
        for (std::size_t i=0;i<shapeCount;++i) if (shapes[i].id==id) return &shapes[i];
        return nullptr;
    };
    const auto unsupported=[&](const char* reason) { error=reason; return false; };
    const auto& terminal=std::get<GeometryData>(value.data);
    std::size_t terminalBytes=0;
    using NativePrefix=NativeBytes<OuterNativeBytes,sizeof(boundaries),sizeof(proxy),sizeof(shapes),
        sizeof(shapeCount),sizeof(geometryCount),sizeof(fieldCount),sizeof(edges),sizeof(meshPayload),
        sizeof(fieldPayload),sizeof(sourcePayload),sizeof(temporaryMesh),sizeof(temporaryField),
        sizeof(displacementCount),sizeof(rootCount),sizeof(constructedCount),sizeof(terminalBytes),
        sizeof(add),sizeof(find),sizeof(unsupported)>;
    static_assert(NativeFits<FactoryNativeFloor,NativePrefix::value,
        NativeProduct<7,sizeof(std::size_t)>::value,sizeof(bool)>::value,
        "Body native event: derive prefix mesh/array/add/checked helpers");
    static_assert(NativeFits<FactoryNativeFloor,NativePrefix::value,sizeof(const char*),sizeof(bool)>::value,
        "Body native event: derive prefix unsupported/status return");
    if (!meshBytes(terminal.positions.size(),terminal.indices.size(),terminalBytes)) return false;
    r.descriptorCopyLogicalBytes=sizeof(ValueDescriptor);
    if (!add(r.descriptorCopyLogicalBytes,terminalBytes-sizeof(GeometryData))
        || !addArrayBytes(value.operations.size(),sizeof(OperationRecord),r.descriptorCopyLogicalBytes)) return false;
    r.sourceComparison.positionElements=terminal.positions.size();
    r.sourceComparison.identityElements=terminal.vertexIds.size();
    r.sourceComparison.indexElements=terminal.indices.size();
    r.sourceComparison.operationRecords=value.operations.size();
    if (!addArrayBytes(value.operations.size(),12,r.sourceComparison.fieldComponents)) return false;
    // Existing retained capacities belong to the exact source closure. Copies
    // below own size-derived arrays; spare capacity chosen by their library is
    // outside the logical recipe, as are allocation node/control-block headers.
    r.sourceLogicalBytes=sizeof(ValueDescriptor);
    if (!addArrayBytes(terminal.positions.capacity(),sizeof(Vec3),r.sourceLogicalBytes)
        || !addArrayBytes(terminal.vertexIds.capacity(),sizeof(StableId),r.sourceLogicalBytes)
        || !addArrayBytes(terminal.indices.capacity(),sizeof(std::uint32_t),r.sourceLogicalBytes)
        || !addArrayBytes(value.operations.capacity(),sizeof(OperationRecord),r.sourceLogicalBytes)
        || !addArrayBytes(value.attributes.capacity(),sizeof(AttributeData),r.sourceLogicalBytes)
        || !addArrayBytes(value.spectrumFields.capacity(),sizeof(spectrum::Binding),r.sourceLogicalBytes)
        || !addArrayBytes(value.scoreFields.capacity(),sizeof(scorefield::Binding),r.sourceLogicalBytes)) return false;

    for (const auto& op:value.operations) {
        if (op.sdfSurface) return unsupported("Body geometry has no SDF source-work schedule");
        if (!op.field.attributeName.empty()) return unsupported("Body geometry has no named-attribute source-work schedule");
        if (op.retainedMesh && (!op.retainedMesh->importedAnimation.empty() || !op.retainedMesh->attributes.empty()))
            return unsupported("Body geometry has no imported-animation or retained-attribute source-work schedule");
        if (!add(r.sourceLogicalBytes,op.field.attributeName.capacity()) || !add(r.sourceLogicalBytes,1)
            || !add(r.descriptorCopyLogicalBytes,op.field.attributeName.size()) || !add(r.descriptorCopyLogicalBytes,1)) return false;
        Shape out; out.id=op.outputStableId;
        const auto* a=find(op.inputStableId); const auto* b=find(op.secondaryInputStableId);
        std::size_t bytes=0;
        using NativeOperation=NativeBytes<NativePrefix::value,sizeof(out),sizeof(a),sizeof(b),sizeof(bytes)>;
        static_assert(NativeFits<FactoryNativeFloor,NativeOperation::value,
            NativeProduct<7,sizeof(std::size_t)>::value,sizeof(bool)>::value,
            "Body native event: derive operation mesh/array/add/checked helpers");
        static_assert(NativeFits<FactoryNativeFloor,NativeOperation::value,
            sizeof(StableId),sizeof(std::size_t),sizeof(const Shape*)>::value,
            "Body native event: derive operation find parameter/counter/return");
        static_assert(NativeFits<FactoryNativeFloor,NativeOperation::value,sizeof(const char*),sizeof(bool)>::value,
            "Body native event: derive operation unsupported/status return");
        switch (op.code) {
        case OperationCode::retainedMesh:
            if (!op.retainedMesh) return unsupported("Body geometry retained source is missing");
            out.vertices=op.retainedMesh->geometry.positions.size(); out.indices=op.retainedMesh->geometry.indices.size();
            if (out.vertices==0 || out.vertices>256 || out.indices==0 || out.indices>384 || out.indices%3!=0)
                return unsupported("Body geometry retained source exceeds its released mesh limits");
            if (!meshBytes(out.vertices,out.indices,bytes) || !add(sourcePayload,bytes+sizeof(RetainedMeshData)-sizeof(GeometryData))) return false;
            if (!add(sourcePayload,1) || !add(r.sourceLogicalBytes,sizeof(RetainedMeshData))
                || !addArrayBytes(op.retainedMesh->geometry.positions.capacity(),sizeof(Vec3),r.sourceLogicalBytes)
                || !addArrayBytes(op.retainedMesh->geometry.vertexIds.capacity(),sizeof(StableId),r.sourceLogicalBytes)
                || !addArrayBytes(op.retainedMesh->geometry.indices.capacity(),sizeof(std::uint32_t),r.sourceLogicalBytes)
                || !addArrayBytes(op.retainedMesh->attributes.capacity(),sizeof(AttributeData),r.sourceLogicalBytes)
                || !add(r.sourceLogicalBytes,op.retainedMesh->importedAnimation.capacity())
                || !add(r.sourceLogicalBytes,1)) return false;
            ++rootCount;
            // Retained source validation and its materialized geometry copy.
            if (!add(r.replayEvaluation.positionElements,out.vertices) || !add(r.replayEvaluation.identityElements,out.vertices)
                || !add(r.replayEvaluation.indexElements,out.indices)) return false;
            if (!add(r.sourceComparison.positionElements,out.vertices) || !add(r.sourceComparison.identityElements,out.vertices)
                || !add(r.sourceComparison.indexElements,out.indices)) return false;
            break;
        case OperationCode::grid: {
            std::uint32_t columns=0,rows=0; std::size_t cells=0;
            static_assert(NativeFits<FactoryNativeFloor,NativeOperation::value,
                sizeof(columns),sizeof(rows),sizeof(cells),sizeof(float),sizeof(std::uint32_t),
                sizeof(float),sizeof(bool)>::value,
                "Body native event: derive Grid exactInteger/finite helper");
            static_assert(NativeFits<FactoryNativeFloor,NativeOperation::value,
                sizeof(columns),sizeof(rows),sizeof(cells),sizeof(std::size_t),sizeof(std::size_t),
                sizeof(bool),sizeof(const char*)>::value,
                "Body native event: derive Grid checked helper/refusal");
            if (!exactIntegerParameter(op.parameters[0],2,columns) || !exactIntegerParameter(op.parameters[1],2,rows)
                || !checkedMul(columns,rows,out.vertices) || !checkedMul(columns-1,rows-1,cells)
                || !checkedMul(cells,6,out.indices)) return unsupported("Body geometry grid shape is invalid");
            ++rootCount;
            // The actual grid vertex loop and six writes in each cell loop.
            if (!add(r.replayEvaluation.positionElements,out.vertices) || !add(r.replayEvaluation.identityElements,out.vertices)
                || !add(r.replayEvaluation.indexElements,out.indices)) return false;
            break;
        }
        case OperationCode::transformTRS:
            if (!a || a->vertices==0) return unsupported("Body geometry transform has no retained mesh shape");
            out.vertices=a->vertices; out.indices=a->indices;
            if (!add(r.replayEvaluation.positionElements,out.vertices) || !add(edges,1)) return false;
            break;
        case OperationCode::positionField:
            if (!a || a->vertices==0) return unsupported("Body geometry position field has no retained mesh shape");
            out.fields=a->vertices;
            // retainedData copies the source; field resize then the identity/position loop.
            if (!meshBytes(a->vertices,a->indices,bytes)) return false;
            temporaryMesh=std::max(temporaryMesh,bytes);
            if (!add(r.replayEvaluation.positionElements,out.fields) || !add(r.replayEvaluation.identityElements,out.fields)
                || !add(r.replayEvaluation.positionElements,out.fields) || !add(r.replayEvaluation.identityElements,out.fields)
                || !add(r.replayEvaluation.indexElements,a->indices)
                || !add(r.replayEvaluation.fieldElements,out.fields) || !add(r.replayEvaluation.fieldElements,out.fields)
                || !addArrayBytes(out.fields,3,r.replayEvaluation.fieldComponents)
                || !add(edges,1)) return false;
            break;
        case OperationCode::fieldVectorMeasure:
        case OperationCode::fieldTimeline:
        case OperationCode::fieldComposeVector:
            if (!a || !a->fields || op.secondaryInputStableId!=0 || op.field.auxiliaryInputStableId!=0
                || op.field.domain!=Domain::vertex
                || (op.code==OperationCode::fieldVectorMeasure && (op.field.mode<3 || op.field.mode>5))
                || (op.code==OperationCode::fieldComposeVector && op.field.mode!=1))
                return unsupported("Body geometry supports unary vertex timeline, axis measure and scaled vector construction only");
            out.fields=a->fields;
            ++constructedCount;
            // assign(count,{}), one evaluation loop, typed input and output components.
            if (!add(r.replayEvaluation.fieldElements,out.fields) || !add(r.replayEvaluation.fieldElements,out.fields)
                || !addArrayBytes(out.fields,op.code==OperationCode::fieldTimeline ? 2 : 4,r.replayEvaluation.fieldComponents)
                || !add(edges,1)) return false;
            // fieldVectorMeasure's three-iteration j loop runs even in axis modes.
            if (op.code==OperationCode::fieldVectorMeasure && !addArrayBytes(out.fields,3,r.replayEvaluation.fieldComponents)) return false;
            break;
        case OperationCode::vectorDisplacement:
            if (!a || !a->vertices || !b || !b->fields || (b->fields!=1 && b->fields!=a->vertices))
                return unsupported("Body geometry displacement has no exact mesh and vector field shapes");
            out.vertices=a->vertices; out.indices=a->indices;
            if (!meshBytes(a->vertices,a->indices,bytes)) return false;
            temporaryMesh=std::max(temporaryMesh,bytes);
            if (!addArrayBytes(out.vertices,sizeof(Vec3),temporaryField)
                || !add(r.replayEvaluation.positionElements,out.vertices) || !add(r.replayEvaluation.positionElements,out.vertices)
                || !addArrayBytes(out.vertices,3,r.replayEvaluation.fieldComponents)
                || !add(edges,2) || !add(displacementCount,1)) return false;
            break;
        case OperationCode::fieldWaterHeight:
            return unsupported("Body geometry has no dynamic-water replay source-work schedule");
        case OperationCode::fieldAudioSpectrum:
        case OperationCode::audioDeformer:
            return unsupported("Body geometry has no spectrum/history replay source-work schedule");
        case OperationCode::fieldScoreSample:
        case OperationCode::evaluateField:
            return unsupported("Body geometry has no canonical Score Field replay source-work schedule");
        default:
            return unsupported("Body geometry operation has no independently derived replay source-work schedule");
        }
        if (out.vertices) {
            if (out.vertices>256 || out.indices>384 || out.indices==0 || out.indices%3!=0
                || !meshBytes(out.vertices,out.indices,bytes) || !add(meshPayload,bytes))
                return unsupported("Body geometry intermediate exceeds its released mesh limits");
            ++geometryCount;
            // Every materialized mesh writes its positions/IDs/indices. TRS copies
            // before its position loop; displacement's retainedData copy is its output.
            if (op.code!=OperationCode::grid && (!add(r.replayEvaluation.positionElements,out.vertices)
                || !add(r.replayEvaluation.identityElements,out.vertices) || !add(r.replayEvaluation.indexElements,out.indices))) return false;
        } else {
            bytes=sizeof(FieldData);
            if (out.fields==0 || out.fields>256 || !addArrayBytes(out.fields,sizeof(FieldElement),bytes)
                || !add(fieldPayload,bytes)) return false;
            temporaryField=std::max(temporaryField,bytes); ++fieldCount;
        }
        shapes[shapeCount++]=out;
    }
    if (!value.attributes.empty()) return unsupported("Body geometry has no terminal-attribute source-work schedule");
    const auto* terminalShape=find(value.stableId);
    static_assert(NativeFits<FactoryNativeFloor,NativePrefix::value,sizeof(terminalShape),
        sizeof(StableId),sizeof(std::size_t),sizeof(const Shape*)>::value,
        "Body native event: derive terminal find parameter/counter/return");
    static_assert(NativeFits<FactoryNativeFloor,NativePrefix::value,sizeof(terminalShape),
        sizeof(const char*),sizeof(bool)>::value,"Body native event: derive terminal refusal");
    if (!terminalShape || terminalShape->vertices!=terminal.positions.size() || terminalShape->indices!=terminal.indices.size())
        return unsupported("Body geometry terminal does not match its derived source shape");

    const auto n=value.operations.size();
    static_assert(NativeFits<FactoryNativeFloor,NativePrefix::value,sizeof(terminalShape),sizeof(n),
        NativeProduct<5,sizeof(std::size_t)>::value,sizeof(bool)>::value,
        "Body native event: derive replay-count/materialized-array helpers");
    // Shared validateOperationPlan scans: orderedInstanceScorePlacementAncestors,
    // main operation loop, vectorDisplacement's complete spectral search, generated
    // output closure check, particleGeometryIntermediatesFit, retained attribute
    // materialization, and the replayed descriptor's operation-record copy.
    if (!add(r.replayEvaluation.operationRecords,n) || !add(r.replayEvaluation.operationRecords,n)
        || !addArrayBytes(displacementCount,n,r.replayEvaluation.operationRecords)
        || !add(r.replayEvaluation.operationRecords,n) || !add(r.replayEvaluation.operationRecords,geometryCount)
        || !add(r.replayEvaluation.operationRecords,geometryCount) || !add(r.replayEvaluation.operationRecords,n)
        || !add(r.replayEvaluation.operationRecords,n) // caller's working descriptor copy
        || !add(r.replayEvaluation.operationRecords,n) // validateParticleGeometry, without intermediate rerun
        // Ancestor insertion visits input edges, dependency closure visits each
        // stored edge plus the root. Duplicate DAG parents remain counted.
        || !add(r.replayEvaluation.dependencyEdges,edges) || !add(r.replayEvaluation.dependencyEdges,edges)
        || !add(r.replayEvaluation.dependencyEdges,1)
        || !addArrayBytes(n,3,r.replayEvaluation.dependencyEdges) // sampled-instance source probes
        || !addArrayBytes(n,4,r.replayEvaluation.fieldComponents) // parameter all_of
        || !addArrayBytes(constructedCount,24,r.replayEvaluation.fieldComponents) // both settings validations, including zeroParameters
        || !addArrayBytes(n-constructedCount,8,r.replayEvaluation.fieldComponents)) return false;

    // Logical map/set entries. Node headers, bucket/tree balancing and spare
    // vector capacity are deliberately outside this source-level storage recipe.
    r.materializedLogicalBytes=sizeof(MaterializedInstancePlan);
    if (!add(r.materializedLogicalBytes,meshPayload) || !add(r.materializedLogicalBytes,fieldPayload)
        || !addArrayBytes(n,sizeof(std::pair<const StableId,CarrierKind>),r.materializedLogicalBytes)
        || !addArrayBytes(n,sizeof(std::pair<const StableId,StableId>),r.materializedLogicalBytes)
        || !addArrayBytes(n,sizeof(std::pair<const StableId,std::set<StableId>>),r.materializedLogicalBytes)
        || !addArrayBytes(edges,sizeof(StableId),r.materializedLogicalBytes)
        || !addArrayBytes(n,sizeof(StableId),r.materializedLogicalBytes) // operationIds
        || !addArrayBytes(n,sizeof(StableId),r.materializedLogicalBytes) // generatedOutputs
        || !addArrayBytes(n,sizeof(StableId),r.materializedLogicalBytes) // closure
        || !addArrayBytes(edges+1,sizeof(StableId),r.materializedLogicalBytes) // pending
        || !addArrayBytes(geometryCount,sizeof(std::pair<const StableId,GeometryData>),r.materializedLogicalBytes)
        || !addArrayBytes(fieldCount,sizeof(std::pair<const StableId,FieldData>),r.materializedLogicalBytes)
        || !addArrayBytes(fieldCount,sizeof(std::pair<const StableId,std::pair<Domain,ValueType>>),r.materializedLogicalBytes)
        || !addArrayBytes(geometryCount,sizeof(std::pair<const StableId,std::vector<AttributeData>>),r.materializedLogicalBytes)
        || !addArrayBytes(geometryCount,sizeof(std::pair<const StableId,std::vector<AttributeData>>),r.materializedLogicalBytes)
        // materialTables has an empty table for each output and each root input.
        || !addArrayBytes(n+rootCount,sizeof(std::pair<const StableId,std::map<std::uint8_t,OperationRecord>>),r.materializedLogicalBytes)
        // orderedInstanceScorePlacementAncestors retains its own parents/edges.
        || !addArrayBytes(n,sizeof(std::pair<const StableId,std::set<StableId>>),r.materializedLogicalBytes)
        || !addArrayBytes(edges,sizeof(StableId),r.materializedLogicalBytes)) return false;
    // Empty vectors, maps and sets still have logical object storage. These
    // match the locals in validateOperationPlan and its ancestor helper.
    const std::size_t controlObjects=sizeof(std::map<StableId,CarrierKind>)
        +sizeof(std::map<StableId,StableId>)+sizeof(std::map<StableId,std::set<StableId>>)
        +sizeof(std::set<StableId>)+sizeof(std::set<StableId>)+sizeof(std::set<StableId>)+sizeof(std::set<StableId>)
        +sizeof(std::map<StableId,InstancesData>)+sizeof(std::map<StableId,FieldData>)
        +sizeof(std::map<StableId,StableId>)+sizeof(std::map<StableId,std::set<StableId>>)
        +sizeof(std::map<StableId,std::pair<Domain,ValueType>>)
        +sizeof(std::map<StableId,std::vector<AttributeData>>)
        +sizeof(std::set<StableId>)+sizeof(std::vector<StableId>)
        +sizeof(std::map<StableId,std::set<StableId>>)+sizeof(std::set<StableId>)+sizeof(std::vector<StableId>);
    if (!add(r.materializedLogicalBytes,controlObjects)) return false;
    static_assert(NativeFits<FactoryNativeFloor,NativePrefix::value,sizeof(terminalShape),sizeof(n),
        sizeof(controlObjects),NativeProduct<5,sizeof(std::size_t)>::value,sizeof(bool)>::value,
        "Body native event: derive control/sample/comparison array/add helpers");
    r.sampleLogicalBytes=terminalBytes;
    // Source comparison, replayed descriptor publication and its retainedData
    // temporary/assignment, then final topology/finite checks and normalization.
    if (!add(r.replayEvaluation.positionElements,r.sourceComparison.positionElements)
        || !add(r.replayEvaluation.identityElements,r.sourceComparison.identityElements)
        || !add(r.replayEvaluation.indexElements,r.sourceComparison.indexElements)
        || !add(r.replayEvaluation.operationRecords,r.sourceComparison.operationRecords)
        || !add(r.replayEvaluation.fieldComponents,r.sourceComparison.fieldComponents)
        || !addArrayBytes(terminal.positions.size(),4,r.replayEvaluation.positionElements)
        || !addArrayBytes(terminal.vertexIds.size(),4,r.replayEvaluation.identityElements)
        || !addArrayBytes(terminal.indices.size(),4,r.replayEvaluation.indexElements)
        || !addArrayBytes(terminal.positions.size(),4,r.replayEvaluation.positionElements)
        || !add(r.replayEvaluation.identityElements,terminal.vertexIds.size())
        || !add(r.replayEvaluation.indexElements,terminal.indices.size())
        // Three prospective sample copies are independent of the two descriptor
        // copies and retainedData copies above. They are not elided by assumption.
        || !addArrayBytes(terminal.positions.size(),3,r.replayEvaluation.positionElements)
        || !addArrayBytes(terminal.vertexIds.size(),3,r.replayEvaluation.identityElements)
        || !addArrayBytes(terminal.indices.size(),3,r.replayEvaluation.indexElements)) return false;
    r.replayBoundaries=boundaries;
    // A future fixed-box sample must compare all three normalized half ratios
    // with the immutable reference. This reserves the comparison, not a PASS.
    if (proxy==ParticleBodyGeometryProxy::fixedBox && !add(r.replayEvaluation.fieldComponents,3)) return false;
    r.retainedSourceSnapshotLogicalBytes=sourcePayload;
    std::size_t visits=0;
    using NativeSuffix=NativeBytes<NativePrefix::value,sizeof(terminalShape),sizeof(n),sizeof(controlObjects),sizeof(visits)>;
    static_assert(NativeFits<FactoryNativeFloor,NativeSuffix::value,
        NativeProduct<12,sizeof(std::size_t)>::value,sizeof(std::initializer_list<std::size_t>),sizeof(bool)>::value,
        "Body native event: derive replay total backing/count/list/checked/status");
    static_assert(NativeFits<FactoryNativeFloor,NativeSuffix::value,
        NativeProduct<5,sizeof(std::size_t)>::value,sizeof(bool)>::value,
        "Body native event: derive final storage array/add/checked/status");
    if (!r.replayEvaluation.total(visits) || !checkedMul(boundaries,visits,r.replayTotalVisits)) return false;
    // A future producer must conserve this prospective envelope: one working
    // descriptor, one replayed descriptor, all materialized intermediates, the
    // retainedData temporary, displacement/field workspace, and three sample
    // owners for producer/publication/returned result. No NRVO is assumed.
    r.replayAttemptLogicalBytes=r.materializedLogicalBytes;
    return addArrayBytes(2,r.descriptorCopyLogicalBytes,r.replayAttemptLogicalBytes)
        && add(r.replayAttemptLogicalBytes,std::max(temporaryMesh,terminalBytes))
        && add(r.replayAttemptLogicalBytes,temporaryField)
        && addArrayBytes(3,r.sampleLogicalBytes,r.replayAttemptLogicalBytes);
}
} // namespace particlebodygeometrydetail

inline bool ParticleBodyGeometryAdmission::deriveScalarCost(
    const ParticleBodyGeometryScalarCensus& scalar,std::size_t boundaries,ParticleBodyGeometryProxy proxy,
    ParticleBodyGeometryScalarCost& result,ParticleBodyGeometryCostRefusal& refusal) noexcept
{
    using namespace videowire::geometry;
    result={}; refusal=ParticleBodyGeometryCostRefusal::none;
    const auto unsupported=[&] { refusal=ParticleBodyGeometryCostRefusal::unsupported; return false; };
    const auto overflow=[&] { refusal=ParticleBodyGeometryCostRefusal::arithmetic; return false; };
    if (!boundaries || !scalar.operationCount || scalar.operationCount>32
        || !scalar.terminalVertices || scalar.terminalVertices>256
        || !scalar.terminalIndices || scalar.terminalIndices>384 || scalar.terminalIndices%3
        || (proxy!=ParticleBodyGeometryProxy::sphere && proxy!=ParticleBodyGeometryProxy::fixedBox)) return unsupported();
    auto& r=result;
    r.operationCount=scalar.operationCount;
    r.terminalVertices=scalar.terminalVertices; r.terminalIndices=scalar.terminalIndices;
    r.nativeDataBytes=sizeof(Data); r.nativeBodyBytes=sizeof(ParticleBodyGeometryAdmission);
    r.censusLogicalBytes=particlebodygeometrydetail::censusLogicalBytes;
    r.factoryDependencyLogicalBytes=sizeof(ParticleBodyGeometryFactoryResources);
    if (!particlebodygeometrydetail::meshBytes(scalar.terminalVertices,scalar.terminalIndices,r.sampleLogicalBytes)) return overflow();
    r.descriptorCopyLogicalBytes=sizeof(ValueDescriptor);
    if (!checkedAdd(r.descriptorCopyLogicalBytes,r.sampleLogicalBytes-sizeof(GeometryData),r.descriptorCopyLogicalBytes)
        || !addArrayBytes(scalar.operationCount,sizeof(OperationRecord),r.descriptorCopyLogicalBytes)
        || !checkedAdd(r.descriptorCopyLogicalBytes,scalar.operationCount,r.descriptorCopyLogicalBytes)) return overflow();
    std::size_t roots=0;
    bool timeline=false;
    if (!checkedMul(scalar.terminalVertices,2,r.publication.positionElements)
        || !checkedMul(scalar.terminalVertices,2,r.publication.identityElements)
        || !checkedMul(scalar.terminalIndices,2,r.publication.indexElements)
        || !checkedMul(scalar.operationCount,2,r.publication.operationRecords)
        || !checkedMul(scalar.operationCount,12,r.publication.fieldComponents)) return overflow();
    for (std::size_t i=0;i<scalar.operationCount;++i) {
        const auto& op=scalar.operations[i];
        if (!op.emptyPayloads) return unsupported();
        switch (op.code) {
        case OperationCode::retainedMesh: {
            if (!op.retainedVertices || op.retainedVertices>256 || !op.retainedIndices
                || op.retainedIndices>384 || op.retainedIndices%3
                || op.vertices!=op.retainedVertices || op.indices!=op.retainedIndices) return unsupported();
            std::size_t mesh=0;
            if (!particlebodygeometrydetail::meshBytes(op.retainedVertices,op.retainedIndices,mesh)
                || !checkedAdd(r.retainedSourceSnapshotLogicalBytes,mesh,r.retainedSourceSnapshotLogicalBytes)
                || !checkedAdd(r.retainedSourceSnapshotLogicalBytes,sizeof(RetainedMeshData)-sizeof(GeometryData),r.retainedSourceSnapshotLogicalBytes)
                || !checkedAdd(r.retainedSourceSnapshotLogicalBytes,1,r.retainedSourceSnapshotLogicalBytes)
                || !checkedAdd(r.publication.positionElements,op.retainedVertices,r.publication.positionElements)
                || !checkedAdd(r.publication.identityElements,op.retainedVertices,r.publication.identityElements)
                || !checkedAdd(r.publication.indexElements,op.retainedIndices,r.publication.indexElements)) return overflow();
            ++r.retainedCount; ++roots;
            break;
        }
        case OperationCode::grid: ++roots; break;
        case OperationCode::transformTRS:
        case OperationCode::positionField:
        case OperationCode::fieldVectorMeasure:
        case OperationCode::fieldComposeVector:
        case OperationCode::vectorDisplacement: break;
        case OperationCode::fieldTimeline: timeline=true; break;
        default: return unsupported();
        }
        if (op.vertices) {
            if (op.vertices>256 || !op.indices || op.indices>384 || op.indices%3 || op.fields) return unsupported();
        } else if (!op.fields || op.fields>256 || op.indices) return unsupported();
        if (op.code!=OperationCode::retainedMesh && (op.retainedVertices || op.retainedIndices)) return unsupported();
    }
    if (roots!=1 || !timeline) return unsupported();
    if (!checkedMul(r.retainedCount,sizeof(RetainedResourcePayload<RetainedMeshData>)-sizeof(RetainedMeshData),r.retainedHolderLogicalBytes)) return overflow();
    // Preserve the released H+S+A+R+X and its subsequent +S+M+C exactly.
    r.retainedConstant=sizeof(ParticleBodyGeometryAdmission)+sizeof(Data)+sizeof(AdmittedValue);
    if (!checkedAdd(r.retainedConstant,r.descriptorCopyLogicalBytes-sizeof(ValueDescriptor),r.retainedConstant)
        || !checkedAdd(r.retainedConstant,r.retainedSourceSnapshotLogicalBytes,r.retainedConstant)
        || !checkedAdd(r.retainedConstant,r.sampleLogicalBytes-sizeof(GeometryData),r.retainedConstant)) return overflow();
    r.factoryAttemptConstant=r.retainedConstant;
    if (!checkedAdd(r.factoryAttemptConstant,r.sampleLogicalBytes,r.factoryAttemptConstant)
        || !checkedAdd(r.factoryAttemptConstant,r.censusLogicalBytes,r.factoryAttemptConstant)) return overflow();
    r.census.operationRecords=scalar.operationCount;
    if (!checkedMul(scalar.operationCount,scalar.operationCount-1,r.census.shapeProbes)
        || !checkedAdd(r.census.shapeProbes,scalar.operationCount,r.census.shapeProbes)) return overflow();
    r.referenceValidation.positionElements=scalar.terminalVertices;
    r.referenceValidation.identityElements=scalar.terminalVertices;
    r.referenceValidation.indexElements=scalar.terminalIndices;
    if (!checkedMul(scalar.terminalVertices,2,r.referenceNormalization.positionElements)) return overflow();
    if (proxy==ParticleBodyGeometryProxy::fixedBox) r.referenceNormalization.fieldComponents=3;
    return true;
}

inline bool ParticleBodyGeometryAdmission::observeOwnedCapacities(const videowire::geometry::AdmittedValue& source,
    ParticleBodyGeometryOwnedCapacityCensus& result,ParticleBodyGeometryCostRefusal& refusal) noexcept
{
    using namespace videowire::geometry;
    result.observedSource=nullptr;
    result.operationCount=0;
    result.terminalVertices=0;
    result.terminalIndices=0;
    result.positions=0;
    result.identities=0;
    result.indices=0;
    result.operations=0;
    result.attributes=0;
    result.spectrumFields=0;
    result.scoreFields=0;
    for (auto& record:result.records) {
        record.attributeName=0;
        record.retained=false;
        record.positions=0;
        record.identities=0;
        record.indices=0;
        record.attributes=0;
        record.importedAnimation=0;
    }
    refusal=ParticleBodyGeometryCostRefusal::unsupported;
    const auto& value=source.descriptor(); const auto* mesh=std::get_if<GeometryData>(&value.data);
    if (value.carrier!=CarrierKind::geometry3D || !mesh || value.operations.empty() || value.operations.size()>32
        || !value.attributes.empty() || !value.spectrumFields.empty() || !value.scoreFields.empty()) return false;
    result.observedSource=&source; result.operationCount=value.operations.size();
    result.terminalVertices=mesh->positions.size(); result.terminalIndices=mesh->indices.size();
    result.positions=mesh->positions.capacity(); result.identities=mesh->vertexIds.capacity(); result.indices=mesh->indices.capacity();
    result.operations=value.operations.capacity(); result.attributes=value.attributes.capacity();
    result.spectrumFields=value.spectrumFields.capacity(); result.scoreFields=value.scoreFields.capacity();
    for (std::size_t i=0;i<value.operations.size();++i) {
        const auto& op=value.operations[i]; auto& record=result.records[i];
        if (op.sdfSurface || !op.field.attributeName.empty()) return false;
        record.attributeName=op.field.attributeName.capacity(); record.retained=bool(op.retainedMesh);
        if (op.retainedMesh) {
            if (op.code!=OperationCode::retainedMesh || !op.retainedMesh->attributes.empty()
                || !op.retainedMesh->importedAnimation.empty()) return false;
            record.positions=op.retainedMesh->geometry.positions.capacity();
            record.identities=op.retainedMesh->geometry.vertexIds.capacity(); record.indices=op.retainedMesh->geometry.indices.capacity();
            record.attributes=op.retainedMesh->attributes.capacity(); record.importedAnimation=op.retainedMesh->importedAnimation.capacity();
        }
    }
    refusal=ParticleBodyGeometryCostRefusal::none;
    return true;
}

inline bool ParticleBodyGeometryAdmission::instantiateScalarCost(const videowire::geometry::AdmittedValue& source,
    const ParticleBodyGeometryScalarCost& affine,const ParticleBodyGeometryOwnedCapacityCensus& capacity,
    ParticleBodyGeometryScalarCost& result,ParticleBodyGeometryCostRefusal& refusal) noexcept
{
    using namespace videowire::geometry;
    refusal=ParticleBodyGeometryCostRefusal::sourceMismatch;
    if (capacity.observedSource!=&source || !capacity.operationCount || capacity.operationCount>32
        || affine.operationCount!=capacity.operationCount || affine.terminalVertices!=capacity.terminalVertices
        || affine.terminalIndices!=capacity.terminalIndices) return false;
    result=affine;
    auto& bytes=result.sourceLogicalBytes; bytes=sizeof(ValueDescriptor);
    const auto overflow=[&] { refusal=ParticleBodyGeometryCostRefusal::arithmetic; return false; };
    if (!addArrayBytes(capacity.positions,sizeof(Vec3),bytes) || !addArrayBytes(capacity.identities,sizeof(StableId),bytes)
        || !addArrayBytes(capacity.indices,sizeof(std::uint32_t),bytes) || !addArrayBytes(capacity.operations,sizeof(OperationRecord),bytes)
        || !addArrayBytes(capacity.attributes,sizeof(AttributeData),bytes) || !addArrayBytes(capacity.spectrumFields,sizeof(spectrum::Binding),bytes)
        || !addArrayBytes(capacity.scoreFields,sizeof(scorefield::Binding),bytes)) return overflow();
    std::size_t retained=0;
    for (std::size_t i=0;i<capacity.operationCount;++i) {
        const auto& record=capacity.records[i];
        if (!checkedAdd(bytes,record.attributeName,bytes) || !checkedAdd(bytes,1,bytes)) return overflow();
        if (record.retained) {
            ++retained;
            if (!checkedAdd(bytes,sizeof(RetainedMeshData),bytes) || !addArrayBytes(record.positions,sizeof(Vec3),bytes)
                || !addArrayBytes(record.identities,sizeof(StableId),bytes) || !addArrayBytes(record.indices,sizeof(std::uint32_t),bytes)
                || !addArrayBytes(record.attributes,sizeof(AttributeData),bytes) || !checkedAdd(bytes,record.importedAnimation,bytes)
                || !checkedAdd(bytes,1,bytes)) return overflow();
        }
    }
    if (retained!=affine.retainedCount) return false;
    result.retainedLogicalBytes=result.retainedConstant;
    result.factoryAttemptLogicalBytes=result.factoryAttemptConstant;
    if (!addArrayBytes(result.retainedSourceCoefficient,bytes,result.retainedLogicalBytes)
        || !addArrayBytes(result.factorySourceCoefficient,bytes,result.factoryAttemptLogicalBytes)) return overflow();
    result.owningFactoryAttemptLogicalBytes=result.factoryAttemptLogicalBytes;
    if (!checkedAdd(result.owningFactoryAttemptLogicalBytes,result.factoryDependencyLogicalBytes,result.owningFactoryAttemptLogicalBytes)
        || !checkedAdd(result.owningFactoryAttemptLogicalBytes,result.retainedHolderLogicalBytes,result.owningFactoryAttemptLogicalBytes)) return overflow();
    refusal=ParticleBodyGeometryCostRefusal::none;
    return true;
}

inline std::shared_ptr<const ParticleBodyGeometryAdmission> admitParticleBodyGeometryWithRetainedResources(
    const std::shared_ptr<const videowire::geometry::AdmittedValue>& source,
    ParticleBodyGeometryProxy proxy,std::size_t boundaries,const ParticleBodyGeometryLimits& limits,
    ParticleBodyGeometryAttempt& attempt,std::string& error,
    const std::shared_ptr<const ParticleBodyCallerLease>& callerResources,
    const videowire::geometry::RetainedResourceOwner& snapshotResources)
{
    using namespace videowire::geometry;
    using namespace particlebodygeometrydetail;
    // CallBase is the existing destination Attempt and Limits. The certificate
    // uses the ordinary omitted-resource union on every inner event: distinct
    // outer proxy/boundaries, empty snapshot argument and default typed owner.
    // Direct retained calls have none of those four forwarding objects. Ordinary
    // explicit-resource calls omit only the default typed owner. References
    // borrow their existing owners and are never measured as guessed ABI slots.
    using Call=NativeBytes<sizeof(attempt),sizeof(limits),sizeof(proxy),sizeof(boundaries),
        sizeof(RetainedResourceOwner),sizeof(std::shared_ptr<const ParticleBodyCallerLease>)>;
    using FactoryNativeFloor=NativeBytes<sizeof(ParticleBodyGeometryAdmission),
        sizeof(ParticleBodyGeometryAdmission::Data),sizeof(AdmittedValue),
        sizeof(ValueDescriptor),sizeof(ValueDescriptor),sizeof(OperationRecord),
        sizeof(OperationRecord),sizeof(OperationRecord),1,1,1,sizeof(GeometryData),censusLogicalBytes>;
    static_assert(NativeFits<FactoryNativeFloor::value,Call::value,sizeof(proxy),sizeof(boundaries),
        sizeof(ParticleBodyGeometryAttempt)>::value,"Body native event: reset RHS Attempt full expression");
    attempt={}; error.clear();
    const auto refuse=[&](const char* reason)->std::shared_ptr<const ParticleBodyGeometryAdmission> { error=reason; return {}; };
    static_assert(NativeFits<FactoryNativeFloor::value,Call::value,sizeof(proxy),sizeof(boundaries),
        sizeof(refuse),sizeof(const char*),sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
        "Body native event: entry refusal reason/returned handle");
    if (!source || (proxy!=ParticleBodyGeometryProxy::sphere && proxy!=ParticleBodyGeometryProxy::fixedBox)
        || !boundaries || boundaries>limits.maxReplayBoundaries) return refuse("Body geometry admission requires an exact source, proxy and bounded replay boundary count");
    const auto& value=source->descriptor(); const auto* mesh=std::get_if<GeometryData>(&value.data);
    if (value.carrier!=CarrierKind::geometry3D || !mesh || value.operations.empty() || value.operations.size()>32
        || mesh->positions.empty() || mesh->positions.size()>256 || mesh->indices.empty() || mesh->indices.size()>384)
        return refuse("Body geometry admission requires the released Geometry3D source limits");
    if (!value.spectrumFields.empty()) return refuse("Body geometry has no spectrum/history replay source-work schedule");
    if (!value.scoreFields.empty()) return refuse("Body geometry has no canonical Score Field replay source-work schedule");
    auto& r=attempt.recipe;
    r.censusLogicalBytes=particlebodygeometrydetail::censusLogicalBytes;
    r.census.operationRecords=value.operations.size();
    // Two complete prior-shape searches per operation, plus terminal lookup.
    if (!checkedMul(value.operations.size(),value.operations.size()-1,r.census.shapeProbes)
        || !checkedAdd(r.census.shapeProbes,value.operations.size(),r.census.shapeProbes))
        return refuse("Body geometry census arithmetic overflow");
    std::size_t census=0;
    using FactoryEarly=NativeBytes<Call::value,sizeof(proxy),sizeof(boundaries),sizeof(refuse),sizeof(mesh),sizeof(census)>;
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryEarly::value,
        NativeProduct<12,sizeof(std::size_t)>::value,sizeof(std::initializer_list<std::size_t>),sizeof(bool)>::value,
        "Body native event: census total backing/count/list/checked/status");
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryEarly::value,sizeof(const char*),
        sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
        "Body native event: census refusal reason/returned handle");
    if (!r.census.total(census) || census>limits.maxCensusVisits || census>limits.maxFactoryVisits
        || r.censusLogicalBytes>limits.maxFactoryAttemptLogicalBytes)
        return refuse("Body geometry census work reservation is short");
    attempt.censusReserved=true;
    if (!particlebodygeometrydetail::derive<FactoryEarly::value,FactoryNativeFloor::value>(value,boundaries,proxy,r,error)) {
        if (error.empty()) error="Body geometry resource arithmetic overflow";
        return {};
    }
    // The new factory's own loops: finite/identity/index validation, bounds,
    // enclosing radius, normalized writes, and snapshot/publication copies.
    r.referenceValidation.positionElements=mesh->positions.size();
    r.referenceValidation.identityElements=mesh->vertexIds.size();
    r.referenceValidation.indexElements=mesh->indices.size();
    if (!checkedMul(mesh->positions.size(),2,r.referenceNormalization.positionElements))
        return refuse("Body geometry normalization arithmetic overflow");
    if (proxy==ParticleBodyGeometryProxy::fixedBox) r.referenceNormalization.fieldComponents=3;
    r.publication=r.sourceComparison;
    if (!checkedAdd(r.publication.positionElements,mesh->positions.size(),r.publication.positionElements)
        || !checkedAdd(r.publication.identityElements,mesh->vertexIds.size(),r.publication.identityElements)
        || !checkedAdd(r.publication.indexElements,mesh->indices.size(),r.publication.indexElements)
        || !checkedAdd(r.publication.operationRecords,value.operations.size(),r.publication.operationRecords))
        return refuse("Body geometry publication arithmetic overflow");
    std::size_t validation=0,normalization=0,publication=0,total=census;
    using FactoryTotals=NativeBytes<FactoryEarly::value,sizeof(validation),sizeof(normalization),sizeof(publication),sizeof(total)>;
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryTotals::value,
        NativeProduct<12,sizeof(std::size_t)>::value,sizeof(std::initializer_list<std::size_t>),sizeof(bool)>::value,
        "Body native event: three sequential reference/publication totals");
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryTotals::value,sizeof(std::size_t),sizeof(std::size_t),
        sizeof(bool),sizeof(const char*),sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
        "Body native event: factory checked sums and reference refusal");
    if (!r.referenceValidation.total(validation) || !r.referenceNormalization.total(normalization)
        || !r.publication.total(publication) || !checkedAdd(total,validation,total)
        || !checkedAdd(total,normalization,total) || !checkedAdd(total,publication,total))
        return refuse("Body geometry factory work arithmetic overflow");
    // Original exact source, independent descriptor/retained-source snapshot,
    // normalized reference and the immutable owner records are all retained.
    r.retainedLogicalBytes=sizeof(ParticleBodyGeometryAdmission)+sizeof(ParticleBodyGeometryAdmission::Data)+sizeof(AdmittedValue);
    if (!checkedAdd(r.retainedLogicalBytes,r.sourceLogicalBytes,r.retainedLogicalBytes)
        || !checkedAdd(r.retainedLogicalBytes,r.descriptorCopyLogicalBytes-sizeof(ValueDescriptor),r.retainedLogicalBytes)
        || !checkedAdd(r.retainedLogicalBytes,r.retainedSourceSnapshotLogicalBytes,r.retainedLogicalBytes)
        || !checkedAdd(r.retainedLogicalBytes,r.sampleLogicalBytes-sizeof(GeometryData),r.retainedLogicalBytes))
        return refuse("Body geometry retained storage arithmetic overflow");
    r.factoryAttemptLogicalBytes=r.retainedLogicalBytes;
    // One prospective owning descriptor and reference transfer remain covered
    // even if a const-member copy occurs. Count failures before publication too.
    if (!checkedAdd(r.factoryAttemptLogicalBytes,r.sourceLogicalBytes,r.factoryAttemptLogicalBytes)
        || !checkedAdd(r.factoryAttemptLogicalBytes,r.sampleLogicalBytes,r.factoryAttemptLogicalBytes)
        || !checkedAdd(r.factoryAttemptLogicalBytes,r.censusLogicalBytes,r.factoryAttemptLogicalBytes)
        || !checkedAdd(r.replayAttemptLogicalBytes,r.retainedLogicalBytes,r.replayAttemptLogicalBytes))
        return refuse("Body geometry attempt storage arithmetic overflow");
    if (total>limits.maxFactoryVisits || r.replayTotalVisits>limits.maxReplayVisits
        || r.retainedLogicalBytes>limits.maxRetainedLogicalBytes
        || r.factoryAttemptLogicalBytes>limits.maxFactoryAttemptLogicalBytes
        || r.replayAttemptLogicalBytes>limits.maxReplayAttemptLogicalBytes)
        return refuse("Body geometry logical storage or named work reservation is short");
    attempt.referenceReserved=true;
    if (mesh->vertexIds.size()!=mesh->positions.size() || mesh->indices.size()%3!=0)
        return refuse("Body geometry reference topology is incomplete");
    // The source already owns admission validity. These no-allocation checks
    // also reject a corrupt retained reference without trusting a digest.
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryTotals::value,sizeof(std::size_t),sizeof(float),sizeof(bool)>::value,
        "Body native event: validation position loop finite helper");
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryTotals::value,sizeof(std::size_t),sizeof(StableId),sizeof(bool)>::value,
        "Body native event: validation identity loop helper");
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryTotals::value,sizeof(std::size_t),sizeof(const char*),
        sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
        "Body native event: validation position/identity loop refusal");
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryTotals::value,sizeof(std::uint32_t),sizeof(const char*),
        sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
        "Body native event: validation index loop refusal");
    for (std::size_t i=0;i<mesh->positions.size();++i)
        if (!finite(mesh->positions[i]) || !validStableId(mesh->vertexIds[i]))
            return refuse("Body geometry reference contains a nonfinite position or invalid identity");
    for (auto index:mesh->indices) if (index>=mesh->positions.size()) return refuse("Body geometry reference index is out of range");
    auto low=mesh->positions.front(),high=low;
    for (const auto& v:mesh->positions) {
        low.x=std::min(low.x,v.x); low.y=std::min(low.y,v.y); low.z=std::min(low.z,v.z);
        high.x=std::max(high.x,v.x); high.y=std::max(high.y,v.y); high.z=std::max(high.z,v.z);
    }
    const Vec3 center{(low.x+high.x)*0.5f,(low.y+high.y)*0.5f,(low.z+high.z)*0.5f};
    float radius=0;
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryTotals::value,sizeof(low),sizeof(high),sizeof(center),sizeof(radius),
        sizeof(decltype(std::hypot(std::hypot(low.x-center.x,low.y-center.y),low.z-center.z))),sizeof(bool)>::value,
        "Body native event: bounds/radius actual hypot result");
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryTotals::value,sizeof(low),sizeof(high),sizeof(center),sizeof(radius),
        sizeof(float),sizeof(bool)>::value,"Body native event: center/radius finite helper");
    for (const auto& v:mesh->positions) radius=std::max(radius,std::hypot(std::hypot(v.x-center.x,v.y-center.y),v.z-center.z));
    if (!finite(center) || !finite(radius) || radius<=1.0e-7f) return refuse("Body geometry reference has no finite enclosing radius");
    const std::array<float,3> half{(high.x-low.x)*0.5f/radius,(high.y-low.y)*0.5f/radius,(high.z-low.z)*0.5f/radius};
    using FactoryReference=NativeBytes<FactoryTotals::value,sizeof(low),sizeof(high),sizeof(center),sizeof(radius),sizeof(half)>;
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryReference::value,
        sizeof(PositiveReferenceHalf),sizeof(float),sizeof(float),sizeof(bool)>::value,
        "Body native event: positive half predicate argument/operator/finite");
    static_assert(NativeFits<FactoryNativeFloor::value,FactoryReference::value,sizeof(const char*),
        sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
        "Body native event: reference/fixedBox/catch refusal");
    // One additive CopySlots(PositiveReferenceHalf,1) owns the real argument's
    // initialization. L keeps its three tests, J its begin/end reads and F its
    // proxy/refusal. Each future W2 must add sizeof(PositiveReferenceHalf) with
    // checked arithmetic before entry. Its actual literal/addition must update
    // Iwork and only the Ifactory sites that evaluate that changed expression;
    // their extents propagate through the existing Ccalc/Z/Wscan/W0 equations.
    if (proxy==ParticleBodyGeometryProxy::fixedBox && !std::all_of(half.begin(),half.end(),PositiveReferenceHalf{}))
        return refuse("Body geometry fixed box requires three positive reference half ratios");
    attempt.publicationReserved=true;
    try {
        auto snapshot=value;
        std::array<std::shared_ptr<const RetainedMeshData>,32> owners{};
        using ReleasedFixed=NativeBytes<sizeof(ParticleBodyGeometryAdmission),sizeof(ParticleBodyGeometryAdmission::Data),
            sizeof(AdmittedValue),sizeof(ValueDescriptor),sizeof(ValueDescriptor),sizeof(GeometryData),censusLogicalBytes>;
        // Admitted+S is the original live source, with no P1 sharing credit.
        // Partial snapshot arrays are bounded by the complete descriptor set.
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners)>::value,
            1,1,0,0>::grid,"Body native Grid event: snapshot copy and owners initialization");
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners)>::value,
            1,1,0,0>::retained,"Body native retained event: snapshot copy and owners initialization");
        for (std::size_t i=0;i<snapshot.operations.size();++i) if (snapshot.operations[i].retainedMesh) {
            // i survives every replacement/helper return and failed construction
            // in this block, and retires before normalized, Data and catch.
            // Allocated RetainedMeshData is in R; the genuine holder increment
            // is separately present in both the event and owning reservation.
            static_assert(NativePayloadFits<ReleasedFixed::value,
                NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(i),
                    sizeof(RetainedResourceOwner),sizeof(std::shared_ptr<RetainedResourcePayload<const RetainedMeshData>>)>::value,
                1,1,1,0>::grid,"Body native Grid event: snapshot holder parameter/construction/failure");
            static_assert(NativePayloadFits<ReleasedFixed::value,
                NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(i),
                    sizeof(RetainedResourceOwner),sizeof(std::shared_ptr<RetainedResourcePayload<const RetainedMeshData>>)>::value,
                1,1,1,0>::retained,"Body native retained event: snapshot holder parameter/construction/failure");
            static_assert(NativePayloadFits<ReleasedFixed::value,
                NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(i),
                    sizeof(std::shared_ptr<RetainedResourcePayload<const RetainedMeshData>>),
                    sizeof(std::shared_ptr<const RetainedMeshData>)>::value,
                1,1,1,0>::grid,"Body native Grid event: snapshot retained helper holder/alias return");
            static_assert(NativePayloadFits<ReleasedFixed::value,
                NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(i),
                    sizeof(std::shared_ptr<RetainedResourcePayload<const RetainedMeshData>>),
                    sizeof(std::shared_ptr<const RetainedMeshData>)>::value,
                1,1,1,0>::retained,"Body native retained event: snapshot retained helper holder/alias return");
            owners[i]=snapshot.operations[i].retainedMesh;
            if (snapshotResources)
                snapshot.operations[i].retainedMesh=makeRetainedResourcePayload<const RetainedMeshData>(snapshotResources,*owners[i]);
            else
                snapshot.operations[i].retainedMesh=std::make_shared<const RetainedMeshData>(*owners[i]);
        }
        GeometryData normalized;
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(normalized),sizeof(Vec3)>::value,
            1,1,1,1>::grid,"Body native Grid event: normalized fill Vec3 argument/failure");
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(normalized),sizeof(Vec3)>::value,
            1,1,1,1>::retained,"Body native retained event: normalized fill Vec3 argument/failure");
        normalized.positions.reserve(mesh->positions.size());
        normalized.vertexIds=mesh->vertexIds; normalized.indices=mesh->indices;
        for (const auto& v:mesh->positions) normalized.positions.push_back({(v.x-center.x)/radius,(v.y-center.y)/radius,(v.z-center.z)/radius});
        // All ten Data inputs bind the already charged live factory objects.
        // Data still owns its complete members; caller headers remain live.
        // This empty sum counts value-parameter objects, not ABI reference slots.
        using DataParameters=NativeBytes<>;
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(normalized),
                sizeof(ParticleBodyGeometryAdmission::Data),DataParameters::value,
                sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission::Data>)>::value,
            1,1,1,1>::grid,"Body native Grid event: Data bound inputs/allocated Data/result/failure");
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(normalized),
                sizeof(ParticleBodyGeometryAdmission::Data),DataParameters::value,
                sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission::Data>)>::value,
            1,1,1,1>::retained,"Body native retained event: Data bound inputs/allocated Data/result/failure");
        auto data=std::make_shared<const ParticleBodyGeometryAdmission::Data>(source,std::move(snapshot),
            std::move(normalized),std::move(owners),center,radius,half,proxy,r,callerResources);
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(normalized),
                sizeof(ParticleBodyGeometryAdmission::Data),sizeof(ParticleBodyGeometryAdmission),sizeof(data),sizeof(data),
                sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
            1,1,1,1>::grid,"Body native Grid event: Admission parameter/local Data/allocated body/failure");
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(normalized),
                sizeof(ParticleBodyGeometryAdmission::Data),sizeof(ParticleBodyGeometryAdmission),sizeof(data),sizeof(data),
                sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
            1,1,1,1>::retained,"Body native retained event: Admission parameter/local Data/allocated body/failure");
        auto admitted=std::shared_ptr<const ParticleBodyGeometryAdmission>(new ParticleBodyGeometryAdmission(std::move(data)));
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(normalized),
                sizeof(ParticleBodyGeometryAdmission::Data),sizeof(ParticleBodyGeometryAdmission),sizeof(data),sizeof(admitted),sizeof(admitted)>::value,
            1,1,1,1>::grid,"Body native Grid event: inner admitted named-return transfer");
        static_assert(NativePayloadFits<ReleasedFixed::value,
            NativeBytes<FactoryReference::value,sizeof(AdmittedValue),sizeof(snapshot),sizeof(owners),sizeof(normalized),
                sizeof(ParticleBodyGeometryAdmission::Data),sizeof(ParticleBodyGeometryAdmission),sizeof(data),sizeof(admitted),sizeof(admitted)>::value,
            1,1,1,1>::retained,"Body native retained event: inner admitted named-return transfer");
        attempt.published=true;
        return admitted;
    } catch (const std::bad_alloc&) {
        return refuse("Body geometry admission allocation failed before publication");
    }
}
inline std::shared_ptr<const ParticleBodyGeometryAdmission> admitParticleBodyGeometry(
    const std::shared_ptr<const videowire::geometry::AdmittedValue>& source,
    ParticleBodyGeometryProxy proxy,std::size_t boundaries,const ParticleBodyGeometryLimits& limits,
    ParticleBodyGeometryAttempt& attempt,std::string& error,
    const std::shared_ptr<const ParticleBodyCallerLease>& callerResources)
{
    using namespace videowire::geometry;
    using namespace particlebodygeometrydetail;
    // The inner frame and all try/derive locals have retired. The empty
    // snapshot argument survives this return expression; default callerResources
    // survives the enclosing caller expression. The returned shared handle is
    // one real result here, with the inner named-return transfer already above.
    static_assert(NativePayloadFits<NativeBytes<sizeof(ParticleBodyGeometryAdmission),
        sizeof(ParticleBodyGeometryAdmission::Data),sizeof(AdmittedValue),sizeof(ValueDescriptor),
        sizeof(ValueDescriptor),sizeof(GeometryData),censusLogicalBytes>::value,
        NativeBytes<sizeof(attempt),sizeof(limits),sizeof(proxy),sizeof(boundaries),sizeof(RetainedResourceOwner),
            sizeof(std::shared_ptr<const ParticleBodyCallerLease>),sizeof(AdmittedValue),
            sizeof(ParticleBodyGeometryAdmission::Data),sizeof(ParticleBodyGeometryAdmission),
            sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
        1,1,1,1>::grid,"Body native Grid event: separate ordinary forwarding return");
    static_assert(NativePayloadFits<NativeBytes<sizeof(ParticleBodyGeometryAdmission),
        sizeof(ParticleBodyGeometryAdmission::Data),sizeof(AdmittedValue),sizeof(ValueDescriptor),
        sizeof(ValueDescriptor),sizeof(GeometryData),censusLogicalBytes>::value,
        NativeBytes<sizeof(attempt),sizeof(limits),sizeof(proxy),sizeof(boundaries),sizeof(RetainedResourceOwner),
            sizeof(std::shared_ptr<const ParticleBodyCallerLease>),sizeof(AdmittedValue),
            sizeof(ParticleBodyGeometryAdmission::Data),sizeof(ParticleBodyGeometryAdmission),
            sizeof(std::shared_ptr<const ParticleBodyGeometryAdmission>)>::value,
        1,1,1,1>::retained,"Body native retained event: separate ordinary forwarding return");
    return admitParticleBodyGeometryWithRetainedResources(source,proxy,boundaries,limits,attempt,error,callerResources,{});
}
} // namespace videorender
