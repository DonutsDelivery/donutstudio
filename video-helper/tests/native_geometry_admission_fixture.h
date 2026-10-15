#pragma once

#include "constructed_field_fixture.h"

inline bool verifyNativeGeometryAdmission(arbitgpu::NativeFixtureSceneBackend& backend,
                                         std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource source(backend);
  const auto capabilities=source.geometryCoreCapabilities();
  if (!capabilities.typedFieldEvaluation
      || capabilities.supportedCarriers!=(carrierBit(CarrierKind::geometry3D)
                                         | carrierBit(CarrierKind::instances3D))
      || capabilities.limits.maxFieldElements<capabilities.limits.maxVertices
      || capabilities.limits.maxFieldElements<capabilities.limits.maxInstances) {
    error="Native Geometry Core must admit bounded mesh and instance fields without advertising field, point or curve draws";
    return false;
  }
  auto value=constructedFieldFixture(1,error);
  if (!value) return false;
  const auto contract=withAttributeContract(constructedFieldMeshContract(),*value);
  const auto admitted=admitValue(*value,contract,error);
  if (!admitted || !decodeRuntimeValue(encodeRuntimeValue(*admitted),contract,
                                       capabilities.limits,{},error)) return false;

  GeometryCorePlanRuntime runtime(source);
  const auto bytes=lowerRuntimePlan(contract,*admitted);
  if (!runtime.admitPreview(bytes,{}, {},error) || !runtime.admitExport(bytes,{}, {},error))
    return false;
  const auto rejectsDraw=[&](const ValueDescriptor& candidate, const PortContract& output) {
    const auto checked=admitValue(candidate,output,error);
    if (!checked) return false;
    const auto payload=lowerRuntimePlan(output,*checked);
    if (runtime.admitPreview(payload,{}, {},error)
        || error.find("does not execute this Geometry Core carrier")==std::string::npos)
      return false;
    return !runtime.admitExport(payload,{}, {},error)
        && error.find("does not execute this Geometry Core carrier")!=std::string::npos;
  };
  const auto mesh=lowerGrid(8501,8501,1,2,2,1,constructedFieldMeshContract(),error);
  if (!mesh) return false;
  PortContract pointsContract;
  pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=4;
  const auto points=lowerPointsFromVertices(*mesh,8502,pointsContract,error);
  PortContract curvesContract;
  curvesContract.carrier=CarrierKind::curves3D;
  curvesContract.maxCurvePoints=8; curvesContract.maxSplines=1;
  const auto curve=lowerCurveCircle(8503,8503,1,8,1,curvesContract,error);
  const auto fieldContract=constructedFieldContract(Domain::vertex,ValueType::vector);
  const auto field=lowerElementField(*mesh,8504,OperationCode::positionField,fieldContract,error);
  if (!points || !curve || !field) return false;
  if (!rejectsDraw(*points,pointsContract) || !rejectsDraw(*curve,curvesContract)
      || !rejectsDraw(*field,fieldContract)) {
    error="Native Geometry Core must reject standalone point, curve and field draws in preview and export";
    return false;
  }
  error.clear(); return true;
}
