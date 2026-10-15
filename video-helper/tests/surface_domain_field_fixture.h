#pragma once

#include "instance_appearance_fixture.h"
#include "../../shared/SurfaceMaterialField.h"

inline std::optional<surfacematerialfield::Binding> surfaceDomainFieldFixture(
    videowire::geometry::Domain domain, std::string& error) {
  using namespace videowire::geometry;
  PortContract curves; curves.carrier=CarrierKind::curves3D;
  curves.maxCurvePoints=16; curves.maxSplines=4; curves.maxAttributes=32;
  PortContract points; points.carrier=CarrierKind::points3D; points.maxPoints=16; points.maxAttributes=32;
  const auto scalar=constructedFieldContract(domain,ValueType::floatValue);
  auto source=domain==Domain::curvePoint || domain==Domain::spline
      ? lowerCurveLine(9801,9801,1,4,2,curves,error)
      : domain==Domain::instance ? instanceAppearanceFixture(0,error)
      : lowerGrid(9801,9801,1,2,2,1,constructedFieldMeshContract(),error);
  if (source && domain==Domain::point) source=lowerPointsFromVertices(*source,9802,points,error);
  if (!source) return std::nullopt;
  FieldOperationSettings settings; settings.domain=domain; settings.valueType=ValueType::floatValue;
  std::optional<ValueDescriptor> field;
  if (domain==Domain::spline) {
    settings.values={0.5,0,0,0,1,0,0,0};
    const auto authored=lowerConstructedField(9803,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,scalar,error);
    const auto stored=authored ? lowerNamedAttribute(*source,&*authored,9804,OperationCode::storeNamedAttribute,
        "value",ValueType::floatValue,curves,error) : std::nullopt;
    if (stored) field=lowerNamedAttribute(*stored,nullptr,9805,OperationCode::readNamedAttribute,"value",ValueType::floatValue,scalar,error);
  } else {
    const auto integer=constructedFieldContract(domain,ValueType::integer);
    const auto index=domain==Domain::face ? lowerFaceIndexField(*source,9803,integer,error)
        : lowerElementField(*source,9803,OperationCode::indexField,integer,error);
    if (index) field=lowerConstructedField(9805,OperationCode::fieldIntegerToFloat,settings,&*index,nullptr,nullptr,scalar,error);
  }
  const auto admitted=field ? admitValue(*field,scalar,error) : std::nullopt;
  if (!admitted) return std::nullopt;
  surfacematerialfield::Binding binding;
  binding.sample.materialNode=9899; binding.sample.target=0; binding.sample.gain=0.25;
  binding.sample.fieldPlan=encodeLoweredPlanText(lowerRuntimePlan(scalar,*admitted));
  if (!surfacematerialfield::valid(binding,error)) return std::nullopt;
  return binding;
}
