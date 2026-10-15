#pragma once

#include "../src/particle_body_replay.h"

inline std::optional<videorender::ParticleParams> particleGeometryFixture(bool animated, std::string& error)
{
    using namespace videowire::geometry;
    PortContract contract; contract.carrier=CarrierKind::geometry3D;
    contract.maxVertices=256; contract.maxIndices=384;
    auto geometry=lowerGrid(9800,9800,1,2,2,1,contract,error);
    if (!geometry) return std::nullopt;
    if (animated)
    {
        PortContract vector; vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex;
        vector.fieldValueType=ValueType::vector; vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=4;
        auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
        auto position=lowerElementField(*geometry,9801,OperationCode::positionField,vector,error);
        FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue; settings.mode=3;
        auto coordinate=position ? lowerConstructedField(9802,OperationCode::fieldVectorMeasure,settings,
            &*position,nullptr,nullptr,scalar,error) : std::nullopt;
        settings.mode=0; settings.values={1,0,0,0,0.25,0,0,0};
        auto time=coordinate ? lowerConstructedField(9803,OperationCode::fieldTimeline,settings,
            &*coordinate,nullptr,nullptr,scalar,error) : std::nullopt;
        settings.mode=1; settings.valueType=ValueType::vector; settings.values={1,0,0,0,0,0,0,0};
        auto offset=time ? lowerConstructedField(9804,OperationCode::fieldComposeVector,settings,
            &*time,nullptr,nullptr,vector,error) : std::nullopt;
        geometry=offset ? lowerVectorDisplacement(*geometry,*offset,9805,1,contract,error) : std::nullopt;
        if (!geometry) return std::nullopt;
    }
    if (!validateParticleGeometry(*geometry,CarrierKind::geometry3D,error)) return std::nullopt;
    auto binding=std::make_shared<videorender::ParticleGeometryBinding>();
    binding->mesh=std::make_shared<const ValueDescriptor>(*geometry); binding->scale=0.4f;
    auto history=std::make_shared<videorender::ParticleHistorySource>(); history->score=std::make_shared<const arbitmod::Score>();
    videorender::ParticleParams p;
    p.motionMode=2; p.historicalReplay=true; p.history=history; p.geometryBinding=binding; p.geometryRevision=1;
    p.geometryCount=1; p.geometryAnchors[0]={0.1f,0.5f,0.1f,0.5f}; p.count=1;
    p.lifetime=1; p.force=0; p.impulseX=0.8f; p.restitution=1; p.bodyRadius=0.015f; p.size=10;
    return p;
}

inline bool prepareParticleGeometryFixture(videorender::ParticleParams& p, double time, std::string& error)
{
    p.preparedBodies.reset(); p.historyProjectSeconds=time;
    const auto state=videorender::replayParticleBodies(p,time,1,nullptr,&error);
    if (!error.empty()) return false;
    p.preparedBodies=std::make_shared<const videorender::ParticleBodyState>(state);
    return true;
}
