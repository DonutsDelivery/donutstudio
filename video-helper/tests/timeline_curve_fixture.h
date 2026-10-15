#pragma once

#include "score_field_fixture.h"

inline videowire::geometry::PortContract timelineContract(videowire::geometry::CarrierKind carrier,
    videowire::geometry::ValueType type=videowire::geometry::ValueType::floatValue) {
  using namespace videowire::geometry;
  PortContract c; c.carrier=carrier; c.overflow=OverflowPolicy::reject;
  if (carrier==CarrierKind::geometry3D) { c.maxVertices=512; c.maxIndices=3072; }
  if (carrier==CarrierKind::curves3D) { c.maxCurvePoints=64; c.maxSplines=1; }
  if (carrier==CarrierKind::points3D) c.maxPoints=64;
  if (carrier==CarrierKind::instances3D) c.maxInstances=64;
  if (carrier==CarrierKind::field) {
    c.fieldDomain=Domain::curvePoint; c.fieldValueType=type; c.fieldInterpolation=Interpolation::constant;
    c.maxFieldElements=64;
  }
  return c;
}

inline std::optional<videowire::geometry::ValueDescriptor> timelineInstanceFixture(std::string& error) {
  using namespace videowire::geometry;
  auto grid=lowerGrid(9801,9801,1,2,2,0.8f,timelineContract(CarrierKind::geometry3D),error);
  auto mesh=lowerMeshGenerator(9802,9802,1,OperationCode::cube,{0.3f,0.3f,0.3f,0},timelineContract(CarrierKind::geometry3D),error);
  auto points=grid ? lowerPointsFromVertices(*grid,9803,timelineContract(CarrierKind::points3D),error) : std::nullopt;
  auto instances=points ? lowerInstanceOnPoints(*points,9802,9804,timelineContract(CarrierKind::instances3D),error) : std::nullopt;
  if (!mesh || !instances) return std::nullopt;
  auto operations=mesh->operations;
  if (!appendUniqueOperations(operations,instances->operations,error)) return std::nullopt;
  instances->operations=std::move(operations); instances->dispatchCount=instances->operations.size();
  auto scalar=timelineContract(CarrierKind::field), integer=scalar, vector=scalar;
  scalar.fieldDomain=integer.fieldDomain=vector.fieldDomain=Domain::instance;
  integer.fieldValueType=ValueType::integer; vector.fieldValueType=ValueType::vector;
  auto indices=lowerElementField(*instances,9805,OperationCode::indexField,integer,error);
  if (!indices) return std::nullopt;
  FieldOperationSettings settings; settings.domain=Domain::instance; settings.valueType=ValueType::floatValue;
  auto values=lowerConstructedField(9806,OperationCode::fieldIntegerToFloat,settings,&*indices,nullptr,nullptr,scalar,error);
  if (!values) return std::nullopt;
  settings.mode=2; settings.values={4,0,0.13,0,0.35,0,0,0};
  auto loop=lowerConstructedField(9807,OperationCode::fieldTimeline,settings,&*values,nullptr,nullptr,scalar,error);
  if (!loop) return std::nullopt;
  settings.mode=1; settings.valueType=ValueType::vector; settings.values={0,1,0,0,0,0,0,0};
  auto offset=lowerConstructedField(9808,OperationCode::fieldComposeVector,settings,&*loop,nullptr,nullptr,vector,error);
  return offset ? lowerVectorDisplacement(*instances,*offset,9809,1,timelineContract(CarrierKind::instances3D),error) : std::nullopt;
}

inline std::optional<videowire::geometry::ValueDescriptor> timelineCurveFixture(std::string& error) {
  using namespace videowire::geometry;
  const auto curves=timelineContract(CarrierKind::curves3D), scalar=timelineContract(CarrierKind::field);
  const auto vector=timelineContract(CarrierKind::field,ValueType::vector);
  auto line=lowerCurveLine(9701,9701,1,4,3,curves,error);
  if (!line) return std::nullopt;
  line=lowerCurveOperation(*line,9714,OperationCode::curveResample,{16,0,0,0},curves,error);
  if (!line) return std::nullopt;
  line=lowerCurveOperation(*line,9715,OperationCode::curveTrim,{0.1f,0.9f,0,0},curves,error);
  if (!line) return std::nullopt;
  auto parameter=lowerElementField(*line,9702,OperationCode::curveParameterField,scalar,error);
  if (!parameter) return std::nullopt;
  FieldOperationSettings settings; settings.domain=Domain::curvePoint; settings.valueType=ValueType::floatValue;
  settings.mode=2; settings.values={4,0,0.5,0,0.25,0,0,0};
  auto loop=lowerConstructedField(9703,OperationCode::fieldTimeline,settings,&*parameter,nullptr,nullptr,scalar,error);
  if (!loop) return std::nullopt;
  settings.valueType=ValueType::vector; settings.mode=1; settings.values={0,1,0,0,0,0,0,0};
  auto offset=lowerConstructedField(9704,OperationCode::fieldComposeVector,settings,&*loop,nullptr,nullptr,vector,error);
  auto displaced=offset ? lowerVectorDisplacement(*line,*offset,9705,1,curves,error) : std::nullopt;
  auto positions=lowerElementField(*line,9706,OperationCode::positionField,vector,error);
  if (!displaced || !positions) return std::nullopt;
  auto tangent=lowerElementField(*displaced,9716,OperationCode::curveTangentField,vector,error);
  settings.values={1,1,1,0,0,0,0,0};
  auto amount=lowerConstructedField(9717,OperationCode::fieldComposeVector,settings,&*loop,nullptr,nullptr,vector,error);
  if (!tangent || !amount) return std::nullopt;
  settings.mode=2;
  auto tangentOffset=lowerConstructedField(9718,OperationCode::fieldVectorMath,settings,&*tangent,&*amount,nullptr,vector,error);
  auto followed=tangentOffset ? lowerVectorDisplacement(*displaced,*tangentOffset,9719,0.4f,curves,error) : std::nullopt;
  if (!followed) return std::nullopt;
  settings.valueType=ValueType::floatValue; settings.mode=1; settings.values={3,0.4,1,0.25,9799,0,0,0};
  auto score=lowerConstructedField(9707,OperationCode::fieldScoreSample,settings,&*positions,nullptr,nullptr,scalar,error);
  auto endpointContract=scalar; endpointContract.fieldValueType=ValueType::boolean;
  auto endpoints=lowerElementField(*line,9720,OperationCode::curveEndpointField,endpointContract,error);
  if (!score || !endpoints) return std::nullopt;
  settings.mode=0; settings.values={0.5,0,0,0,0,0,0,0};
  auto tapered=lowerConstructedField(9721,OperationCode::fieldSelect,settings,&*score,nullptr,&*endpoints,scalar,error);
  auto radius=tapered ? lowerCurveOperation(*followed,9708,OperationCode::curveSetRadius,{1,1,0,0},curves,error,&*tapered) : std::nullopt;
  if (!radius) return std::nullopt;
  settings.mode=3; settings.values={3,0.4,0,0.25,9799,0,0,0};
  auto links=lowerConstructedField(9709,OperationCode::fieldScoreSample,settings,&*positions,nullptr,nullptr,scalar,error);
  auto tilt=links ? lowerCurveOperation(*radius,9710,OperationCode::curveSetTilt,{0,0,0,0},curves,error,&*links) : std::nullopt;
  if (!tilt) return std::nullopt;
  settings.mode=1; settings.valueType=ValueType::vector; settings.values={0,0,0.2,0,0,0,0,0};
  auto noteOffset=lowerConstructedField(9712,OperationCode::fieldComposeVector,settings,&*score,nullptr,nullptr,vector,error);
  auto notePosition=noteOffset ? lowerVectorDisplacement(*tilt,*noteOffset,9713,1,curves,error) : std::nullopt;
  return notePosition ? lowerCurveOperation(*notePosition,9711,OperationCode::curveToTube,{0.09f,6,0,0},
      timelineContract(CarrierKind::geometry3D),error) : std::nullopt;
}

template <typename ReadPixels>
bool verifyNativeTimelineCurves(arbitgpu::NativeFixtureSceneBackend& backend,ReadPixels readPixels,std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend); GeometryCorePlanRuntime runtime(capabilities);
  auto grid=timelineInstanceFixture(error);
  auto gridValue=grid ? admitValue(*grid,timelineContract(CarrierKind::instances3D),error) : std::nullopt;
  if (!gridValue) return false;
  const auto gridBytes=lowerRuntimePlan(timelineContract(CarrierKind::instances3D),*gridValue);
  const auto gridPreview=runtime.admitPreview({1,1,1,9800,1,PlanUse::preview},gridBytes,{}, {},error);
  const auto gridExport=runtime.admitExport({1,1,1,9800,1,PlanUse::exportRender},gridBytes,{}, {},error);
  if (!gridPreview || !gridExport) return false;
  const auto g0=executeNativeGeometry(backend,*gridPreview,64,64,error,false,{},nullptr,{},0);
  const auto g1=executeNativeGeometry(backend,*gridPreview,64,64,error,false,{},nullptr,{},1);
  const auto gx=executeNativeGeometry(backend,*gridExport,64,64,error,false,{},nullptr,{},1);
  if (!g0 || !g1 || !gx) return false;
  if (readPixels(g0->frame)==readPixels(g1->frame) || readPixels(g1->frame)!=readPixels(gx->frame)
      || g0->drawnInstanceIds!=g1->drawnInstanceIds || g1->drawnInstanceIds!=gx->drawnInstanceIds) {
    error="Grid loop must animate native instance transforms and preserve identities across preview and export"; return false;
  }
  auto fixture=timelineCurveFixture(error);
  auto admitted=fixture ? admitValue(*fixture,timelineContract(CarrierKind::geometry3D),error) : std::nullopt;
  if (!admitted) return false;
  auto bytes=lowerRuntimePlan(timelineContract(CarrierKind::geometry3D),*admitted);
  auto preview=runtime.admitPreview({1,1,1,9700,1,PlanUse::preview},bytes,{}, {},error);
  auto exported=runtime.admitExport({1,1,1,9700,1,PlanUse::exportRender},bytes,{}, {},error);
  if (!preview || !exported) return false;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.canonicalBlockCFrame=scoreFieldFrame();
  auto first=executeNativeGeometry(backend,*preview,64,64,error,false,inputs,nullptr,{},0);
  auto later=executeNativeGeometry(backend,*preview,64,64,error,false,inputs,nullptr,{},1);
  auto looped=executeNativeGeometry(backend,*preview,64,64,error,false,inputs,nullptr,{},4);
  inputs.canonicalBlockCFrame=scoreFieldFrame(true);
  auto offline=executeNativeGeometry(backend,*exported,64,64,error,false,inputs,nullptr,{},1);
  auto sought=executeNativeGeometry(backend,*preview,64,64,error,false,inputs,nullptr,{},0);
  if (!first || !later || !looped || !offline || !sought) return false;
  const auto a=readPixels(first->frame),b=readPixels(later->frame);
  if (a.empty() || a==b || a!=readPixels(looped->frame) || a!=readPixels(sought->frame)
      || b!=readPixels(offline->frame)) {
    error="Timeline curve pixels must animate, close the loop, reproduce seeks and match reordered-score export"; return false;
  }
  inputs.canonicalBlockCFrame=scoreFieldFrame(false,true);
  auto silent=executeNativeGeometry(backend,*preview,64,64,error,false,inputs,nullptr,{},0);
  if (!silent || readPixels(silent->frame)==a) {
    error="Empty score must use the authored radius and tilt bias"; return false;
  }
  return true;
}
