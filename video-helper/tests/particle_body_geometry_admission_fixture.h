#pragma once

#include "../src/particle_body_geometry_admission.h"
#include "../../shared/GeometryCoreTransport.h"

static_assert(!std::is_default_constructible_v<videorender::ParticleBodyGeometryAdmission>,
    "Body geometry admission cannot be manufactured from public default fields");

inline videowire::geometry::PortContract particleBodyGeometryFixtureContract()
{
    videowire::geometry::PortContract contract;
    contract.carrier=videowire::geometry::CarrierKind::geometry3D;
    contract.maxVertices=256; contract.maxIndices=384;
    return contract;
}

// Literal pyramid, independent of renderer normalization and mesh generators.
inline videowire::geometry::ValueDescriptor particleBodyPyramidDescriptor(float tipX=0)
{
    using namespace videowire::geometry;
    GeometryData mesh;
    mesh.positions={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{tipX,0,1}};
    mesh.vertexIds={11,19,23,41,67};
    mesh.indices={0,2,1,0,3,2,0,1,4,1,2,4,2,3,4,3,0,4};
    auto retained=std::make_shared<RetainedMeshData>(); retained->geometry=mesh;
    ValueDescriptor value; value.carrier=CarrierKind::geometry3D;
    value.stableId=83001; value.sourceStableId=83000; value.sourceRevision=7;
    value.data=mesh; value.dispatchCount=1;
    OperationRecord op{83001,OperationCode::retainedMesh,83000,83001,{}};
    op.retainedMesh=std::move(retained); value.operations={std::move(op)};
    return value;
}

inline std::shared_ptr<const videowire::geometry::AdmittedValue> particleBodyFixtureAdmitted(
    const videowire::geometry::ValueDescriptor& value,std::string& error)
{
    using namespace videowire::geometry;
    auto admitted=admitValue(value,particleBodyGeometryFixtureContract(),error);
    if (!admitted) return {};
    // Use the released lowered/decoded authority too, not a fabricated receipt.
    const auto lowered=lowerRuntimePlan(particleBodyGeometryFixtureContract(),*admitted);
    auto bytes=decodeLoweredRuntimePlan(lowered,error);
    auto reopened=bytes ? decodeRuntimeValue(bytes->runtimeValue,bytes->contract,{}, {},error) : std::nullopt;
    return reopened ? std::make_shared<const AdmittedValue>(*reopened) : nullptr;
}

inline bool verifyParticleBodyGeometryAdmissionOracles(std::string& error)
{
    using namespace videowire::geometry;
    using namespace videorender;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    const auto close=[](float a,float b) { return std::abs(a-b)<2.0e-6f; };
    ParticleBodyGeometryLimits limits;
    ParticleBodyGeometryAttempt attempt;
    const auto literal=particleBodyPyramidDescriptor();
    const auto source=particleBodyFixtureAdmitted(literal,error);
    if (!source) return fail("Body admission literal source did not pass existing admission/transport");
    const auto sourceMeshOwner=source->descriptor().operations.front().retainedMesh;
    const auto original=source->descriptor();
    const auto body=admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::fixedBox,121,limits,attempt,error);
    if (!body || !attempt.published || !attempt.censusReserved || !attempt.referenceReserved || !attempt.publicationReserved)
        return fail("Body admission did not publish the literal fixed-box reference");
    const float invRootThree=0.5773502691896257645f;
    const auto& mesh=body->referencePrototype();
    if (body->source()!=source || body->sourceReceipt().deterministicDigest!=source->receipt().deterministicDigest
        || body->terminalIdentity()!=83001 || body->sourceIdentity()!=83000 || body->structuralRevision()!=7
        || !equal(body->referenceCenter(),Vec3{}) || !close(body->referenceRadius(),1.7320508075688772935f)
        || mesh.vertexIds!=std::vector<StableId>{11,19,23,41,67}
        || mesh.indices!=std::vector<std::uint32_t>{0,2,1,0,3,2,0,1,4,1,2,4,2,3,4,3,0,4}
        || !close(mesh.positions[0].x,-invRootThree) || !close(mesh.positions[0].y,-invRootThree)
        || !close(mesh.positions[0].z,-invRootThree) || !close(mesh.positions[4].x,0)
        || !close(mesh.positions[4].z,invRootThree)
        || !std::all_of(body->referenceBoxHalfRatios().begin(),body->referenceBoxHalfRatios().end(),
            [&](float v) { return close(v,invRootThree); }))
        return fail("Body admission pyramid normalization or proxy ratios disagree with the independent literal");
    if (body->ownedDescriptor().operations.front().retainedMesh==sourceMeshOwner
        || !equal(body->ownedDescriptor().operations.front().retainedMesh->geometry,sourceMeshOwner->geometry)
        || body->recipe().sourceLogicalBytes==body->sourceReceipt().bufferBytes
        || body->recipe().replayBoundaries!=121 || body->recipe().retainedLogicalBytes==0
        || body->recipe().replayAttemptLogicalBytes<=body->recipe().retainedLogicalBytes)
        return fail("Body admission did not retain independent source bytes and independently derived resource accounting");
    // Independent literal loop census for the one retained five-vertex source.
    // It includes exact-source comparison, shared replay validation/copies,
    // normalization and the three prospective sample owner transfers.
    const auto& work=body->recipe().replayEvaluation;
    if (work.operationRecords!=9 || work.dependencyEdges!=4 || work.positionElements!=75
        || work.identityElements!=60 || work.indexElements!=216 || work.fieldComponents!=27
        || body->recipe().replayTotalVisits!=121*391)
        return fail("Body admission retained pyramid replay schedule disagrees with the literal loop census");
    std::size_t comparison=0;
    if (!body->recipe().sourceComparison.total(comparison)
        || !body->matchesExactSource(source,comparison) || body->matchesExactSource(source,comparison-1))
        return fail("Body exact-source comparison did not reserve its own named work");
    const auto equivalent=particleBodyFixtureAdmitted(literal,error);
    if (!equivalent || equivalent->receipt().deterministicDigest!=source->receipt().deterministicDigest
        || body->matchesExactSource(equivalent,comparison))
        return fail("Body admission accepted a matching digest as exact source ownership");
    const auto copy=*body;
    if (copy.source()!=source || &copy.referencePrototype()!=&body->referencePrototype())
        return fail("Body admission copy expanded its shared immutable prototype");

    // A different authentic source changes appearance. These are new admissions,
    // not caller-manufactured evaluated samples or evidence of replay ticks.
    const auto movedSource=particleBodyFixtureAdmitted(particleBodyPyramidDescriptor(0.5f),error);
    const auto moved=movedSource ? admitParticleBodyGeometry(movedSource,ParticleBodyGeometryProxy::sphere,121,limits,attempt,error) : nullptr;
    if (!moved || !close(moved->referencePrototype().positions[4].x,0.2886751345948128823f)
        || !close(moved->referenceRadius(),1.7320508075688772935f)
        || moved->referencePrototype().vertexIds!=mesh.vertexIds || moved->referencePrototype().indices!=mesh.indices)
        return fail("Body admission moved-tip literal changed conserved topology or normalization");
    Transform transform; transform.translation={8,-4,2}; transform.scale={2,2,2};
    const auto transformed=lowerTransformGeometry(literal,83002,transform,particleBodyGeometryFixtureContract(),error);
    const auto transformedSource=transformed ? particleBodyFixtureAdmitted(*transformed,error) : nullptr;
    const auto translated=transformedSource ? admitParticleBodyGeometry(transformedSource,ParticleBodyGeometryProxy::fixedBox,121,limits,attempt,error) : nullptr;
    if (!translated || translated->referencePrototype().vertexIds!=mesh.vertexIds || translated->referencePrototype().indices!=mesh.indices)
        return fail("Body admission uniform transform lost exact topology");
    for (std::size_t i=0;i<mesh.positions.size();++i)
        if (!close(translated->referencePrototype().positions[i].x,mesh.positions[i].x)
            || !close(translated->referencePrototype().positions[i].y,mesh.positions[i].y)
            || !close(translated->referencePrototype().positions[i].z,mesh.positions[i].z))
            return fail("Body admission uniform translation/scale changed the normalized reference");

    // Genuine existing lowering for a topology-stable timeline field chain.
    auto grid=lowerGrid(83100,83100,7,2,2,1,particleBodyGeometryFixtureContract(),error);
    PortContract vector; vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex;
    vector.fieldValueType=ValueType::vector; vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=4;
    auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
    auto position=grid ? lowerElementField(*grid,83101,OperationCode::positionField,vector,error) : std::nullopt;
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue; settings.mode=3;
    auto coordinate=position ? lowerConstructedField(83102,OperationCode::fieldVectorMeasure,settings,&*position,nullptr,nullptr,scalar,error) : std::nullopt;
    settings.mode=0; settings.values={1,0,0,0,0.25,0,0,0};
    auto time=coordinate ? lowerConstructedField(83103,OperationCode::fieldTimeline,settings,&*coordinate,nullptr,nullptr,scalar,error) : std::nullopt;
    settings.mode=1; settings.valueType=ValueType::vector; settings.values={1,0,0,0,0,0,0,0};
    auto offset=time ? lowerConstructedField(83104,OperationCode::fieldComposeVector,settings,&*time,nullptr,nullptr,vector,error) : std::nullopt;
    auto dynamic=offset ? lowerVectorDisplacement(*grid,*offset,83105,1,particleBodyGeometryFixtureContract(),error) : std::nullopt;
    const auto dynamicSource=dynamic ? particleBodyFixtureAdmitted(*dynamic,error) : nullptr;
    const auto dynamicBody=dynamicSource ? admitParticleBodyGeometry(dynamicSource,ParticleBodyGeometryProxy::sphere,3,limits,attempt,error) : nullptr;
    if (!dynamicBody || !hasRuntimeFields(dynamicBody->ownedDescriptor())
        || dynamicBody->recipe().replayEvaluation.fieldElements!=32
        || dynamicBody->recipe().census.operationRecords!=6 || dynamicBody->recipe().census.shapeProbes!=36)
        return fail("Body admission timeline chain did not retain its independently derived field/census schedule");
    std::size_t perBoundary=0;
    if (!dynamicBody->recipe().replayEvaluation.total(perBoundary)
        || dynamicBody->recipe().replayTotalVisits!=3*perBoundary)
        return fail("Body admission replay work used a presentation FPS instead of the explicit boundary count");

    const auto retainedRecipe=body->recipe();
    auto shortLimits=limits; shortLimits.maxRetainedLogicalBytes=retainedRecipe.retainedLogicalBytes-1;
    if (admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::fixedBox,121,shortLimits,attempt,error)
        || attempt.referenceReserved || attempt.publicationReserved || attempt.published)
        return fail("Body admission short retained capacity executed reference work");
    shortLimits=limits; shortLimits.maxReplayVisits=retainedRecipe.replayTotalVisits-1;
    if (admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::fixedBox,121,shortLimits,attempt,error)
        || attempt.referenceReserved || attempt.publicationReserved || attempt.recipe.replayTotalVisits!=retainedRecipe.replayTotalVisits)
        return fail("Body admission short prospective replay work executed reference work or lost its failure recipe");
    shortLimits=limits; shortLimits.maxFactoryAttemptLogicalBytes=0;
    if (admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::sphere,121,shortLimits,attempt,error) || attempt.censusReserved)
        return fail("Body admission short census storage executed the census");
    shortLimits=limits; shortLimits.maxCensusVisits=0;
    if (admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::sphere,121,shortLimits,attempt,error) || attempt.censusReserved)
        return fail("Body admission short census work executed the census");
    std::size_t censusWork=0,referenceWork=0,normalizationWork=0,publicationWork=0;
    if (!retainedRecipe.census.total(censusWork) || !retainedRecipe.referenceValidation.total(referenceWork)
        || !retainedRecipe.referenceNormalization.total(normalizationWork) || !retainedRecipe.publication.total(publicationWork))
        return fail("Body admission fixture's named factory work overflowed");
    shortLimits=limits; shortLimits.maxFactoryVisits=censusWork+referenceWork+normalizationWork+publicationWork-1;
    if (admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::fixedBox,121,shortLimits,attempt,error)
        || attempt.referenceReserved || attempt.publicationReserved)
        return fail("Body admission short full factory work executed reference work");
    shortLimits=limits; shortLimits.maxFactoryAttemptLogicalBytes=retainedRecipe.factoryAttemptLogicalBytes-1;
    if (admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::fixedBox,121,shortLimits,attempt,error)
        || attempt.referenceReserved || attempt.publicationReserved)
        return fail("Body admission short prospective owner transfer storage executed reference work");
    shortLimits=limits; shortLimits.maxReplayAttemptLogicalBytes=retainedRecipe.replayAttemptLogicalBytes-1;
    if (admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::fixedBox,121,shortLimits,attempt,error)
        || attempt.referenceReserved || attempt.publicationReserved)
        return fail("Body admission short materialized replay storage executed reference work");
    shortLimits=limits; shortLimits.maxReplayBoundaries=std::numeric_limits<std::size_t>::max();
    if (admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::sphere,std::numeric_limits<std::size_t>::max(),shortLimits,attempt,error)
        || attempt.referenceReserved || error.find("overflow")==std::string::npos)
        return fail("Body admission replay arithmetic overflow reached reference work");

    auto degenerate=literal;
    auto collapsed=std::make_shared<RetainedMeshData>(*degenerate.operations.front().retainedMesh);
    for (auto& p:collapsed->geometry.positions) p={0,0,0};
    degenerate.data=collapsed->geometry; degenerate.operations.front().retainedMesh=collapsed;
    const auto flatSource=particleBodyFixtureAdmitted(degenerate,error);
    if (!flatSource || admitParticleBodyGeometry(flatSource,ParticleBodyGeometryProxy::sphere,1,limits,attempt,error)
        || attempt.publicationReserved) return fail("Body admission degenerate radius published a reference");
    auto malformed=literal; std::get<GeometryData>(malformed.data).indices[0]=999;
    if (particleBodyFixtureAdmitted(malformed,error)) return fail("Body admission fixture bypassed malformed-source authority refusal");
    malformed=literal; std::get<GeometryData>(malformed.data).positions[0].x=std::numeric_limits<float>::infinity();
    if (particleBodyFixtureAdmitted(malformed,error)) return fail("Body admission fixture bypassed nonfinite-source authority refusal");

    // Existing static mesh generators still admit; this new schedule refuses
    // their unaccounted materializer instead of changing the released API.
    const auto cube=lowerMeshGenerator(83200,83200,7,OperationCode::cube,{1,1,1,0},particleBodyGeometryFixtureContract(),error);
    const auto cubeSource=cube ? particleBodyFixtureAdmitted(*cube,error) : nullptr;
    if (!cubeSource || admitParticleBodyGeometry(cubeSource,ParticleBodyGeometryProxy::sphere,1,limits,attempt,error)
        || attempt.referenceReserved || error.find("source-work schedule")==std::string::npos)
        return fail("Body admission unsupported schedule altered static acceptance or reached reference work");
    const auto water=grid ? lowerWaterSurfaceSimulation(*grid,83301,83302,1,2,0,0,particleBodyGeometryFixtureContract(),error) : std::nullopt;
    const auto waterSource=water ? particleBodyFixtureAdmitted(*water,error) : nullptr;
    if (!waterSource || admitParticleBodyGeometry(waterSource,ParticleBodyGeometryProxy::sphere,1,limits,attempt,error)
        || attempt.referenceReserved || error.find("dynamic-water")==std::string::npos)
        return fail("Body admission dynamic water schedule altered existing admission or reached reference work");
    auto aliasedLiteral=particleBodyPyramidDescriptor();
    const auto mutableProducer=std::const_pointer_cast<RetainedMeshData>(aliasedLiteral.operations.front().retainedMesh);
    const auto aliasedSource=particleBodyFixtureAdmitted(aliasedLiteral,error);
    const auto frozen=aliasedSource ? admitParticleBodyGeometry(aliasedSource,ParticleBodyGeometryProxy::sphere,1,limits,attempt,error) : nullptr;
    mutableProducer->geometry.positions[4].x=0.75f;
    if (!frozen || !close(frozen->referencePrototype().positions[4].x,0)
        || !close(frozen->ownedDescriptor().operations.front().retainedMesh->geometry.positions[4].x,0)
        || !frozen->matchesExactSource(aliasedSource,std::numeric_limits<std::size_t>::max()))
        return fail("Body admission retained a producer's mutable geometry alias");
    if (!equal(std::get<GeometryData>(source->descriptor().data),std::get<GeometryData>(original.data))
        || source->descriptor().operations.front().retainedMesh!=sourceMeshOwner
        || !body->matchesExactSource(source,comparison))
        return fail("Body admission failure changed the prior exact source or published owner");
    error.clear(); return true;
}
