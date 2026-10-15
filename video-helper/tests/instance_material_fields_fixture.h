#pragma once

#include "instance_appearance_fixture.h"
#include "score_field_fixture.h"
#include "../../shared/GeometryCoreScene.h"

inline std::optional<videowire::geometry::ValueDescriptor> instanceMaterialFieldsFixture(
    const std::string& property, bool table, bool score, std::string& error) {
  using namespace videowire::geometry;
  auto instances=instanceAppearanceFixture(0,error);
  if (!instances) return std::nullopt;
  const auto contract=appearanceInstanceContract();
  const auto scalar=constructedFieldContract(Domain::instance,ValueType::floatValue);
  const auto integer=constructedFieldContract(Domain::instance,ValueType::integer);
  const auto color=constructedFieldContract(Domain::instance,ValueType::color);
  const auto boolean=constructedFieldContract(Domain::instance,ValueType::boolean);
  auto indices=lowerElementField(*instances,9501,OperationCode::indexField,integer,error);
  FieldOperationSettings settings; settings.domain=Domain::instance; settings.valueType=ValueType::floatValue;
  auto coordinates=indices ? lowerConstructedField(9502,OperationCode::fieldIntegerToFloat,settings,
      &*indices,nullptr,nullptr,scalar,error) : std::nullopt;
  if (!coordinates) return std::nullopt;
  settings.mode=1; settings.values={4,0,0.05,0,0.8,0.1,0,0};
  auto amount=lowerConstructedField(9503,OperationCode::fieldTimeline,settings,
      &*coordinates,nullptr,nullptr,scalar,error);
  if (score) {
    auto positions=lowerElementField(*instances,9504,OperationCode::positionField,
        constructedFieldContract(Domain::instance,ValueType::vector),error);
    settings.values={4,0.7,0.1,0.25,9599,0,0,0};
    amount=positions ? lowerConstructedField(9503,OperationCode::fieldScoreSample,settings,
        &*positions,nullptr,nullptr,scalar,error) : std::nullopt;
  }
  settings.valueType=ValueType::boolean; settings.mode=0; settings.values={2,0,0,0,0,0,0,0};
  auto selected=lowerConstructedField(9505,OperationCode::fieldCompare,settings,
      &*coordinates,nullptr,nullptr,boolean,error);
  if (!amount || !selected) return std::nullopt;
  if (property=="color" || property=="emission") {
    settings.valueType=ValueType::color; settings.values={0,0,0,1,1,0,0,0};
    auto low=lowerConstructedField(9506,OperationCode::fieldConstant,settings,
        nullptr,nullptr,nullptr,color,error);
    settings.values={0,1,0,1,0,0,0,0};
    amount=low ? lowerConstructedField(9507,OperationCode::fieldMix,settings,
        &*low,nullptr,&*amount,color,error) : std::nullopt;
  } else if (property=="materialIndex") {
    settings.valueType=ValueType::integer; settings.values={};
    // The loop selects slot zero at t=0 and slot one at t=3.
    settings.valueType=ValueType::floatValue; settings.mode=2; settings.values={2,0,0,0,0,0,0,0};
    auto doubled=lowerConstructedField(9508,OperationCode::fieldMath,settings,
        &*amount,nullptr,nullptr,scalar,error);
    settings.valueType=ValueType::integer; settings.mode=0; settings.values={};
    amount=doubled ? lowerConstructedField(9509,OperationCode::fieldFloatToInteger,settings,
        &*doubled,nullptr,nullptr,integer,error) : std::nullopt;
  }
  if (!amount) return std::nullopt;
  auto value=property=="materialIndex" ? lowerMaterialIndices(*instances,*amount,9510,contract,error)
      : lowerInstanceAppearance(*instances,*amount,&*selected,9510,property,contract,error);
  if (table || property=="materialIndex") {
    const std::array<double,8> base{0.45,0.4,0.35,0,0,0,0.4,0.6};
    value=value ? lowerMaterialTableSlot(*value,9511,9521,0,base,1,contract,error) : std::nullopt;
    if (property=="materialIndex") {
      const std::array<double,8> green{0.05,1,0.05,0,0.3,0,0,0.6};
      value=value ? lowerMaterialTableSlot(*value,9512,9522,1,green,1,contract,error) : std::nullopt;
    }
  }
  return value;
}

template <typename ReadPixels>
bool verifyNativeInstanceMaterialFields(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  for (const auto* property : {"color","emission","opacity","metallic","roughness","materialIndex"})
    for (const bool table : {false,true}) {
      const auto value=instanceMaterialFieldsFixture(property,table,false,error);
      if (!value) return false;
      const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
      const auto admitted=admitValue(*value,contract,error);
      if (!admitted) return false;
      GeometryCorePlanRuntime runtime(capabilities);
      const auto bytes=lowerRuntimePlan(contract,*admitted);
      const auto preview=runtime.admitPreview({1,1,1,9510,1,PlanUse::preview},bytes,{}, {},error);
      const auto exported=runtime.admitExport({1,1,1,9510,1,PlanUse::exportRender},bytes,{}, {},error);
      if (!preview || !exported) return false;
      const auto draw=[&](const RuntimeAdmission& plan,double time) {
        return executeNativeGeometry(backend,plan,64,64,error,false,{},nullptr,{},time);
      };
      const auto first=draw(*preview,0),later=draw(*preview,3),offline=draw(*exported,3),sought=draw(*preview,0);
      if (!first || !later || !offline || !sought) return false;
      const auto a=readPixels(first->frame),b=readPixels(later->frame);
      if (a.empty() || a==b || b!=readPixels(offline->frame) || a!=readPixels(sought->frame)
          || first->drawnInstanceIds!=later->drawnInstanceIds
          || first->stats.vertexBytes!=4*sizeof(HarmonicMIDI::grid::SceneVertex)) {
        error=std::string("Instance ")+property+" must change pixels with time, preserve shared vertices and reproduce export and seeks";
        return false;
      }
    }
  const auto score=instanceMaterialFieldsFixture("emission",false,true,error);
  if (!score) return false;
  const auto contract=withAttributeContract(appearanceInstanceContract(),*score);
  const auto admitted=admitValue(*score,contract,error);
  if (!admitted) return false;
  GeometryCorePlanRuntime runtime(capabilities);
  const auto bytes=lowerRuntimePlan(contract,*admitted);
  const auto preview=runtime.admitPreview({1,1,1,9510,1,PlanUse::preview},bytes,{}, {},error);
  const auto exported=runtime.admitExport({1,1,1,9510,1,PlanUse::exportRender},bytes,{}, {},error);
  if (!preview || !exported) return false;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.canonicalBlockCFrame=scoreFieldFrame();
  const auto notes=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
  inputs.canonicalBlockCFrame=scoreFieldFrame(true);
  const auto offline=executeNativeGeometry(backend,*exported,64,64,error,false,inputs);
  inputs.canonicalBlockCFrame=scoreFieldFrame(false,true);
  const auto empty=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
  if (!notes || !offline || !empty) return false;
  if (readPixels(notes->frame)!=readPixels(offline->frame)
      || readPixels(notes->frame)==readPixels(empty->frame)) {
    error="Score-driven instance emission must use canonical note data in preview and export"; return false;
  }
  return true;
}
