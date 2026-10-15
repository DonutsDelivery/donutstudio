#pragma once

#include "particle_body_geometry_admission_fixture.h"
#include "../src/particle_body_geometry_sample.h"

static_assert(!std::is_default_constructible_v<videorender::ParticleBodyGeometrySample>
    && !std::is_copy_constructible_v<videorender::ParticleBodyGeometrySample>,
    "Evaluated body samples require the private factory and share through immutable owners");

inline std::optional<videowire::geometry::ValueDescriptor> particleBodyTimelinePyramidDescriptor(
    std::string& error,bool overflowAppearance=false)
{
    using namespace videowire::geometry;
    // Base z=-1 produces phase 0, tip z=1 produces phase 1/4.
    // At t=0 the source tip -1/2 plus displacement 2*(1/4) is 0.
    // At t=3/4, tip phase wraps to 0 while base phase is 3/4.
    const auto mesh=particleBodyPyramidDescriptor(overflowAppearance ? 0:-0.5f);
    PortContract vector; vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex;
    vector.fieldValueType=ValueType::vector; vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=5;
    auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
    auto position=lowerElementField(mesh,84002,OperationCode::positionField,vector,error);
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue; settings.mode=5;
    auto coordinate=position ? lowerConstructedField(84003,OperationCode::fieldVectorMeasure,settings,&*position,nullptr,nullptr,scalar,error) : std::nullopt;
    settings.mode=overflowAppearance ? 0:1;
    settings.values=overflowAppearance ? std::array<double,8>{1,0,0,0,1,0,0,0}
        : std::array<double,8>{1,0,0.125,0.125,1,0,0,0};
    auto time=coordinate ? lowerConstructedField(84004,OperationCode::fieldTimeline,settings,&*coordinate,nullptr,nullptr,scalar,error) : std::nullopt;
    settings.mode=1; settings.valueType=ValueType::vector; settings.values={1,0,0,0,0,0,0,0};
    auto offset=time ? lowerConstructedField(84005,OperationCode::fieldComposeVector,settings,&*time,nullptr,nullptr,vector,error) : std::nullopt;
    return offset ? lowerVectorDisplacement(mesh,*offset,84006,
        overflowAppearance ? std::numeric_limits<float>::max():2,
        particleBodyGeometryFixtureContract(),error) : std::nullopt;
}

inline bool verifyParticleBodyGeometrySampleOracles(std::string& error)
{
    using namespace videowire::geometry;
    using namespace videorender;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    const auto close=[](float a,float b) { return std::abs(a-b)<2.0e-6f; };
    const std::vector<StableId> ids{11,19,23,41,67};
    const std::vector<std::uint32_t> indices{0,2,1,0,3,2,0,1,4,1,2,4,2,3,4,3,0,4};
    ParticleBodyGeometryLimits admissionLimits;
    ParticleBodyGeometryAttempt admissionAttempt;
    ParticleBodyGeometrySampleLimits limits;
    ParticleBodyGeometrySampleAttempt attempt;
    const auto value=particleBodyTimelinePyramidDescriptor(error);
    const auto source=value ? particleBodyFixtureAdmitted(*value,error):nullptr;
    const auto sphere=source ? admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::sphere,32,admissionLimits,admissionAttempt,error):nullptr;
    const auto sampler=sphere ? makeParticleBodyGeometrySampler(sphere,limits,attempt,error):nullptr;
    if (!sampler || !attempt.derivationReserved || attempt.evaluationReserved || !hasRuntimeFields(sphere->ownedDescriptor()))
        return fail("Body sampler did not derive a genuine released timeline source");
    // Independent source-loop census. The accepted envelope's 837 visits plus
    // 162 newly reserved visits cover typed zero/parameter checks, unique-ID insertion,
    // normalization, topology/finite loops and half-ratio checks. Derivation
    // reads six records and reserves 36 shape probes. Three nine-counter
    // sums add 27 control visits, and two add 18 per sample. No PASS implied.
    const auto& recipe=sampler->recipe();
    const auto& work=recipe.admittedEvaluation;
    const auto& extra=recipe.incrementalEvaluation;
    if (work.operationRecords!=52 || work.dependencyEdges!=31 || work.positionElements!=100
        || work.identityElements!=75 || work.indexElements!=252 || work.fieldElements!=40
        || work.fieldComponents!=287 || extra.positionElements!=20 || extra.identityElements!=15
        || extra.indexElements!=36 || extra.fieldComponents!=91 || recipe.controlVisits!=18
        || recipe.derivationControlVisits!=27 || recipe.perAttemptVisits!=1017
        || recipe.totalVisits!=32*1017+69 || recipe.derivation.operationRecords!=6
        || recipe.derivation.shapeProbes!=36)
        return fail("Body sampler source work differs from the independent six-operation census");
    const auto initial=sampler->sample(source,7,0,attempt,error);
    if (!initial || !attempt.evaluationReserved || !attempt.published || initial->admission()!=sphere
        || initial->source()!=source || initial->structuralRevision()!=7 || initial->absoluteTick()!=0
        || initial->ticksPerSecond!=28224000
        || initial->sourceReceipt().deterministicDigest!=source->receipt().deterministicDigest
        || initial->prototype().vertexIds!=ids || initial->prototype().indices!=indices)
        return fail("Body sampler did not retain the exact source, receipt, rational tick and topology");
    const float invRootThree=0.5773502691896257645f;
    const std::array<Vec3,5> zero{{{-invRootThree,-invRootThree,-invRootThree},
        {invRootThree,-invRootThree,-invRootThree},{invRootThree,invRootThree,-invRootThree},
        {-invRootThree,invRootThree,-invRootThree},{0,0,invRootThree}}};
    const auto checkPositions=[&](const ParticleBodyGeometrySample& sample,const auto& expected) {
        if (sample.prototype().positions.size()!=expected.size()) return false;
        for (std::size_t i=0;i<expected.size();++i) {
            const auto& a=sample.prototype().positions[i]; const auto& b=expected[i];
            if (!close(a.x,b.x) || !close(a.y,b.y) || !close(a.z,b.z)) return false;
        }
        return true;
    };
    if (!checkPositions(*initial,zero) || !equal(initial->center(),Vec3{})
        || !close(initial->radius(),1.7320508075688772935f))
        return fail("Body sampler zero-time normalization differs from literal pyramid positions");
    // These dyadic phases translate the complete mesh. Normalization deliberately
    // removes their common translation. The 1/240-second tick exercises a real
    // fractional replay boundary without assuming adjacent outputs are distinct.
    for (const auto pair:std::array<std::pair<std::int64_t,float>,3>{{{7056000,0.5f},{14112000,1},{117600,0.008333333333333333f}}}) {
        const auto sample=sampler->sample(source,7,pair.first,attempt,error);
        if (!sample || sample->absoluteTick()!=pair.first || !close(sample->center().x,pair.second)
            || !close(sample->center().y,0) || !close(sample->center().z,0)
            || !checkPositions(*sample,zero) || sample->prototype().vertexIds!=ids || sample->prototype().indices!=indices)
            return fail("Body sampler fractional boundary changed literal translation-normalized appearance or topology");
    }
    // At 3/4 seconds raw base x values are 1/2 and 5/2; tip x=-1/2.
    // AABB center is (1,0,0), enclosing radius sqrt(17)/2, half ratios
    // (3/sqrt(17),2/sqrt(17),2/sqrt(17)). These constants are independent
    // of the production bounds/radius helpers and evaluated result.
    const float inverseRadius=0.4850712500726659470f;
    const std::array<Vec3,5> changed{{{-0.2425356250363329735f,-inverseRadius,-inverseRadius},
        {0.7276068751089989205f,-inverseRadius,-inverseRadius},
        {0.7276068751089989205f,inverseRadius,-inverseRadius},
        {-0.2425356250363329735f,inverseRadius,-inverseRadius},
        {-0.7276068751089989205f,0,inverseRadius}}};
    const auto changedSample=sampler->sample(source,7,21168000,attempt,error);
    const auto laterCycle=sampler->sample(source,7,49392000,attempt,error);
    const auto seek=sampler->sample(source,7,0,attempt,error);
    const auto repeated=sampler->sample(source,7,21168000,attempt,error);
    const auto heldTerminal=sampler->sample(source,7,21168000,attempt,error);
    if (!changedSample || !laterCycle || !seek || !repeated || !heldTerminal
        || !checkPositions(*changedSample,changed) || !checkPositions(*laterCycle,changed)
        || !checkPositions(*repeated,changed) || !checkPositions(*heldTerminal,changed)
        || !checkPositions(*seek,zero) || changedSample->absoluteTick()!=21168000
        || laterCycle->absoluteTick()!=49392000 || repeated->absoluteTick()!=21168000
        || heldTerminal->absoluteTick()!=21168000 || !equal(changedSample->center(),Vec3{1,0,0})
        || !close(changedSample->radius(),2.0615528128088302749f)
        || !close(changedSample->boxHalfRatios()[0],0.7276068751089989205f)
        || !close(changedSample->boxHalfRatios()[1],inverseRadius)
        || !close(changedSample->boxHalfRatios()[2],inverseRadius)
        || changedSample->prototype().vertexIds!=ids || changedSample->prototype().indices!=indices)
        return fail("Body sampler changed-tip, absolute repeat, seek or held-terminal literals disagree");
    // This is exact-tick appearance evidence only. Actual replay window selection
    // and body state/contact conservation require the later production join.
    const auto box=admitParticleBodyGeometry(source,ParticleBodyGeometryProxy::fixedBox,4,admissionLimits,admissionAttempt,error);
    const auto boxSampler=box ? makeParticleBodyGeometrySampler(box,limits,attempt,error):nullptr;
    auto boxInitial=boxSampler ? boxSampler->sample(source,7,0,attempt,error):nullptr;
    const auto boxTranslate=boxSampler ? boxSampler->sample(source,7,7056000,attempt,error):nullptr;
    if (!boxInitial || !boxTranslate || !checkPositions(*boxTranslate,zero)
        || boxSampler->sample(source,7,21168000,attempt,error) || !attempt.evaluationReserved || attempt.published
        || error.find("half ratios")==std::string::npos || !checkPositions(*boxInitial,zero)
        || boxSampler->usage().attempts!=3 || boxSampler->usage().visits!=69+3*boxSampler->recipe().perAttemptVisits
        || boxSampler->usage().attemptLogicalBytes!=0 || boxSampler->usage().reservedRetainedLogicalBytes!=0)
        return fail("Body sampler fixed-box ratio refusal did not conserve incumbent or account failed work");

    const auto equivalent=particleBodyFixtureAdmitted(*value,error);
    const auto usageBefore=sampler->usage();
    if (!equivalent || equivalent->receipt().deterministicDigest!=source->receipt().deterministicDigest
        || sampler->sample(equivalent,7,0,attempt,error) || attempt.evaluationReserved
        || sampler->sample(source,8,0,attempt,error) || attempt.evaluationReserved
        || sampler->sample(source,7,-1,attempt,error) || attempt.evaluationReserved
        || sampler->sample(source,7,std::numeric_limits<std::int64_t>::max(),attempt,error) || attempt.evaluationReserved
        || sampler->usage().attempts!=usageBefore.attempts)
        return fail("Body sampler token, source-revision or unrepresentable-tick refusal reached guarded work");
    // Genuine released admission owns const retained sources. Same-token source
    // corruption cannot be produced through its public API. Do not cast away
    // that constness to claim executed coverage of the defensive frozen guard.
    auto changedValue=*value;
    auto changedOwner=std::make_shared<RetainedMeshData>(*changedValue.operations.front().retainedMesh);
    changedOwner->geometry.positions[4].x=0.25f;
    changedValue.operations.front().retainedMesh=changedOwner;
    std::get<GeometryData>(changedValue.data).positions[4].x=0.75f;
    const auto changedSource=particleBodyFixtureAdmitted(changedValue,error);
    auto topologyValue=*value;
    auto topologyOwner=std::make_shared<RetainedMeshData>(*topologyValue.operations.front().retainedMesh);
    topologyOwner->geometry.indices[0]=4;
    topologyValue.operations.front().retainedMesh=topologyOwner;
    std::get<GeometryData>(topologyValue.data).indices[0]=4;
    const auto topologySource=particleBodyFixtureAdmitted(topologyValue,error);
    if (!changedSource || !topologySource || sampler->sample(changedSource,7,0,attempt,error)
        || attempt.evaluationReserved || sampler->sample(topologySource,7,0,attempt,error)
        || attempt.evaluationReserved || !checkPositions(*initial,zero))
        return fail("Body sampler authentic changed-tip/topology source substitution changed incumbent");
    auto malformed=*value;
    std::get<GeometryData>(malformed.data).positions[4].x=std::numeric_limits<float>::infinity();
    if (particleBodyFixtureAdmitted(malformed,error))
        return fail("Body sampler nonfinite fixture bypassed the released source admission");
    malformed=*value; std::get<GeometryData>(malformed.data).indices[0]=999;
    if (particleBodyFixtureAdmitted(malformed,error))
        return fail("Body sampler topology fixture bypassed the released source admission");
    Transform transform; transform.translation={8,-4,2}; transform.scale={2,2,2};
    const auto transformedValue=lowerTransformGeometry(*value,84007,transform,particleBodyGeometryFixtureContract(),error);
    const auto transformedSource=transformedValue ? particleBodyFixtureAdmitted(*transformedValue,error):nullptr;
    const auto transformedBody=transformedSource ? admitParticleBodyGeometry(transformedSource,ParticleBodyGeometryProxy::sphere,2,admissionLimits,admissionAttempt,error):nullptr;
    const auto transformedSampler=transformedBody ? makeParticleBodyGeometrySampler(transformedBody,limits,attempt,error):nullptr;
    const auto transformedZero=transformedSampler ? transformedSampler->sample(transformedSource,7,0,attempt,error):nullptr;
    const auto transformedTip=transformedSampler ? transformedSampler->sample(transformedSource,7,21168000,attempt,error):nullptr;
    if (!transformedZero || !transformedTip || !checkPositions(*transformedZero,zero)
        || !checkPositions(*transformedTip,changed) || !equal(transformedZero->center(),Vec3{8,-4,2})
        || !equal(transformedTip->center(),Vec3{10,-4,2})
        || !close(transformedZero->radius(),3.464101615137754587f)
        || !close(transformedTip->radius(),4.1231056256176605498f))
        return fail("Body sampler uniform translation/scale changed the literal normalized appearance");
    const auto explosiveValue=particleBodyTimelinePyramidDescriptor(error,true);
    const auto explosiveSource=explosiveValue ? particleBodyFixtureAdmitted(*explosiveValue,error):nullptr;
    const auto explosiveBody=explosiveSource ? admitParticleBodyGeometry(explosiveSource,ParticleBodyGeometryProxy::sphere,2,admissionLimits,admissionAttempt,error):nullptr;
    const auto explosiveSampler=explosiveBody ? makeParticleBodyGeometrySampler(explosiveBody,limits,attempt,error):nullptr;
    const auto finiteInitial=explosiveSampler ? explosiveSampler->sample(explosiveSource,7,0,attempt,error):nullptr;
    if (!finiteInitial || explosiveSampler->sample(explosiveSource,7,56448000,attempt,error)
        || !attempt.evaluationReserved || attempt.published || error.find("finite")==std::string::npos
        || !checkPositions(*finiteInitial,zero) || explosiveSampler->usage().attempts!=2
        || explosiveSampler->usage().attemptLogicalBytes!=0)
        return fail("Body sampler genuine timeline overflow did not refuse after admitted evaluation with incumbent conserved");

    auto shortLimits=limits; shortLimits.maxDerivationVisits=68;
    if (makeParticleBodyGeometrySampler(sphere,shortLimits,attempt,error) || attempt.derivationReserved)
        return fail("Body sampler short derivation work executed its census");
    shortLimits=limits; shortLimits.maxRetainedLogicalBytes=recipe.retainedBaseLogicalBytes+recipe.retainedSampleLogicalBytes-1;
    if (makeParticleBodyGeometrySampler(sphere,shortLimits,attempt,error) || attempt.evaluationReserved)
        return fail("Body sampler one-byte short sample owner reached evaluation");
    shortLimits=limits; shortLimits.maxAttemptLogicalBytes=recipe.retainedBaseLogicalBytes+recipe.attemptLogicalBytes-1;
    if (makeParticleBodyGeometrySampler(sphere,shortLimits,attempt,error) || attempt.evaluationReserved)
        return fail("Body sampler one-byte short attempt reached evaluation");
    shortLimits=limits; shortLimits.maxTotalVisits=recipe.totalVisits-1;
    if (makeParticleBodyGeometrySampler(sphere,shortLimits,attempt,error) || attempt.evaluationReserved)
        return fail("Body sampler one-visit short full schedule reached evaluation");
    shortLimits=limits;
    shortLimits.maxRetainedLogicalBytes=recipe.retainedBaseLogicalBytes+recipe.retainedSampleLogicalBytes;
    shortLimits.maxAttemptLogicalBytes=recipe.retainedBaseLogicalBytes+recipe.retainedSampleLogicalBytes+recipe.attemptLogicalBytes;
    auto bounded=makeParticleBodyGeometrySampler(sphere,shortLimits,attempt,error);
    auto owner=bounded ? bounded->sample(source,7,0,attempt,error):nullptr;
    auto escaped=owner;
    if (!owner || bounded->usage().retainedLogicalBytes!=shortLimits.maxRetainedLogicalBytes)
        return fail("Body sampler escaped-owner fixture did not retain one admitted sample");
    owner.reset();
    if (bounded->sample(source,7,7056000,attempt,error) || attempt.evaluationReserved
        || bounded->usage().attempts!=1 || !checkPositions(*escaped,zero))
        return fail("Body sampler credited a still-escaped sample as released");
    const auto escapedPrototype=&escaped->prototype(); auto handle=escaped;
    if (&handle->prototype()!=escapedPrototype) return fail("Body sampler handle duplicated its shared appearance prototype");
    escaped.reset(); handle.reset();
    if (bounded->usage().retainedLogicalBytes!=recipe.retainedBaseLogicalBytes
        || !bounded->sample(source,7,7056000,attempt,error) || !attempt.published
        || bounded->usage().attempts!=2 || bounded->usage().attemptLogicalBytes!=0)
        return fail("Body sampler final escaped release did not restore the logical allowance");
    // Keeping a result alive also keeps its source ledger alive after the sampler
    // handle is released. No latest-sample slot is the resource authority.
    auto survivor=bounded->sample(source,7,0,attempt,error); bounded.reset();
    if (!survivor || survivor->source()!=source || !checkPositions(*survivor,zero))
        return fail("Body sampler result lost its exact source after sampler release");
    const auto staticSource=particleBodyFixtureAdmitted(particleBodyPyramidDescriptor(),error);
    const auto oneBody=staticSource ? admitParticleBodyGeometry(staticSource,ParticleBodyGeometryProxy::sphere,1,admissionLimits,admissionAttempt,error):nullptr;
    const auto oneSampler=oneBody ? makeParticleBodyGeometrySampler(oneBody,limits,attempt,error):nullptr;
    const auto one=oneSampler ? oneSampler->sample(staticSource,7,0,attempt,error):nullptr;
    if (!one || oneSampler->sample(staticSource,7,0,attempt,error) || attempt.evaluationReserved
        || oneSampler->usage().attempts!=1 || !checkPositions(*one,zero))
        return fail("Body sampler exceeded the admitted attempt count or lost incumbent on work shortage");
    const auto cube=lowerMeshGenerator(84200,84200,7,OperationCode::cube,{1,1,1,0},particleBodyGeometryFixtureContract(),error);
    const auto cubeSource=cube ? particleBodyFixtureAdmitted(*cube,error):nullptr;
    if (!cubeSource || admitParticleBodyGeometry(cubeSource,ParticleBodyGeometryProxy::sphere,1,admissionLimits,admissionAttempt,error)
        || admissionAttempt.referenceReserved || makeParticleBodyGeometrySampler({},limits,attempt,error)
        || attempt.derivationReserved)
        return fail("Body sampler unsupported source bypassed its released admission prerequisite");
    error.clear(); return true;
}
