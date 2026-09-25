#pragma once

#include "constructed_field_fixture.h"
#include "instance_appearance_fixture.h"
#include "../../shared/GeometryAudioDeformer.h"

inline std::optional<videowire::geometry::ValueDescriptor> orderedAudioGeometryFixture(std::string& error)
{
    using namespace videowire::geometry;
    auto mesh=lowerMeshGenerator(9600,9600,1,OperationCode::plane,{1.8f,1.8f,0,0},constructedFieldMeshContract(),error);
    spectrum::Binding binding; binding.firstBand=binding.lastBand=0; binding.gain=0.6f;
    auto deformed=mesh ? lowerAudioDeformer(*mesh,9601,binding,constructedFieldMeshContract(),error) : std::nullopt;
    PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D;
    pointsContract.overflow=OverflowPolicy::reject; pointsContract.maxPoints=16;
    auto points=deformed ? lowerPointDistribution(*deformed,9602,OperationCode::pointsOnFaces,12,17,pointsContract,error) : std::nullopt;
    auto shape=lowerMeshGenerator(9603,9603,1,OperationCode::cube,{0.12f,0.12f,0.12f,0},constructedFieldMeshContract(),error);
    auto instances=points && shape ? lowerInstanceOnPoints(*points,shape->stableId,9604,appearanceInstanceContract(),error) : std::nullopt;
    if (!instances) return std::nullopt;
    auto operations=shape->operations;
    if (!appendUniqueOperations(operations,instances->operations,error)) return std::nullopt;
    instances->operations=std::move(operations); instances->dispatchCount=instances->operations.size();
    auto position=lowerElementField(*instances,9605,OperationCode::positionField,
        constructedFieldContract(Domain::instance,ValueType::vector),error);
    FieldOperationSettings settings; settings.domain=Domain::instance; settings.valueType=ValueType::floatValue;
    settings.mode=2; settings.values={0,0,0,0,0.6,0,0,0};
    auto height=position ? lowerConstructedField(9606,OperationCode::fieldGradient,settings,&*position,nullptr,nullptr,
        constructedFieldContract(Domain::instance,ValueType::floatValue),error) : std::nullopt;
    settings.valueType=ValueType::color; settings.mode=0; settings.values={0.1,0.8,0.9,1,1,0,0,0};
    auto color=lowerConstructedField(9607,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,
        constructedFieldContract(Domain::instance,ValueType::color),error);
    settings.values={1,0.05,0.1,1,0.5,0,0,0};
    auto mixed=height && color ? lowerConstructedField(9608,OperationCode::fieldMix,settings,&*color,nullptr,&*height,
        constructedFieldContract(Domain::instance,ValueType::color),error) : std::nullopt;
    return mixed ? lowerInstanceAppearance(*instances,*mixed,nullptr,9609,"color",appearanceInstanceContract(),error) : std::nullopt;
}

template <typename ReadPixels>
bool verifyNativeOrderedAudioGeometry(arbitgpu::NativeFixtureSceneBackend& backend,
                                     ReadPixels readPixels,std::string& error)
{
    using namespace videowire::geometry;
    for (int analysisSource=0;analysisSource<3;++analysisSource) {
        auto value=orderedAudioGeometryFixture(error);
        if (!value) return false;
        value->spectrumFields.front().source=static_cast<spectrum::Source>(analysisSource);
        value->spectrumFields.front().sourceTrackId=analysisSource==0 ? -1 : 71;
        const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
        const auto admitted=admitValue(*value,contract,error);
        if (!admitted) return false;
        videohelper::geometry::NativeGeometryCoreCapabilitySource capabilities(backend);
        videohelper::geometry::GeometryCorePlanRuntime runtime(capabilities);
        const auto bytes=lowerRuntimePlan(contract,*admitted);
        const auto preview=runtime.admitPreview({1,1,1,9609,1,videohelper::geometry::PlanUse::preview},bytes,{}, {},error);
        const auto exportPlan=runtime.admitExport({1,1,1,9609,1,videohelper::geometry::PlanUse::exportRender},bytes,{}, {},error);
        if (!preview || !exportPlan) return false;
        videohelper::geometry::SpectrumEvaluation input;
        float selectedEnergy=0;
        input.sourceFeaturesAt=[&selectedEnergy,analysisSource](spectrum::Source source,std::int32_t track,
            double,spectrum::Bands& bands) {
            if (static_cast<int>(source)!=analysisSource || track!=71) return false;
            bands={}; bands[0]=selectedEnergy; return true;
        };
        input.timeSeconds=0.5;
        auto silent=videohelper::geometry::executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
        if (!silent) return false;
        const auto silentPixels=readPixels(silent->frame);
        if (analysisSource==0) input.bands[0]=1;
        else selectedEnergy=1;
        auto active=videohelper::geometry::executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
        if (!active) return false;
        const auto activePixels=readPixels(active->frame);
        auto exportInput=input; exportInput.followers.clear();
        auto exported=videohelper::geometry::executeNativeGeometry(backend,*exportPlan,64,64,error,false,{},&exportInput);
        if (!exported) return false;
        input.bands[0]=0;
        selectedEnergy=0;
        auto sought=videohelper::geometry::executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
        if (!sought) return false;
        if (activePixels==silentPixels || activePixels!=readPixels(exported->frame)
            || silentPixels!=readPixels(sought->frame) || active->drawnInstanceIds!=silent->drawnInstanceIds
            || active->stats.submittedInstanceCount!=12 || active->stats.instancedDrawCount!=1) {
            error="Ordered audio surface instances lost native appearance, seek/export parity or shared mesh instancing";
            return false;
        }
        if (analysisSource!=0) {
            input.sourceFeaturesAt={};
            if (videohelper::geometry::executeNativeGeometry(backend,*preview,64,64,error,false,{},&input)
                || error.find("Selected track/group spectrum is unavailable")==std::string::npos) {
                error="Ordered geometry replay silently substituted master for a missing selected source";
                return false;
            }
            error.clear();
        }
    }
    return true;
}
