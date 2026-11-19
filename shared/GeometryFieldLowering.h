#pragma once

// GeometryCore.h includes this header after its own definitions, and #pragma once ends the
// cycle: whichever of the two is reached first, the other finds it already open.
#include "GeometryCore.h"
#include "GeometryPortContract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace videowire::geometry {
inline std::optional<ValueDescriptor> lowerMaterialTableSlot(const ValueDescriptor& source,
    StableId valueId, StableId materialId, std::uint8_t slot, const std::array<double,8>& pbr,
    float opacity, PortContract contract, std::string& error) {
  if ((source.carrier!=CarrierKind::geometry3D && source.carrier!=CarrierKind::instances3D)
      || source.operations.size()>=kMaximumOperations) {
    error="Material table requires bounded Geometry3D or Instances3D"; return std::nullopt;
  }
  auto result=source; result.stableId=valueId;
  FieldOperationSettings settings; settings.domain=elementDomain(source.carrier); settings.valueType=ValueType::color;
  settings.mode=slot; settings.values=pbr;
  result.operations.push_back({valueId,OperationCode::materialTableSlot,source.stableId,valueId,
      {opacity,0,0,0},materialId,{},settings});
  result.dispatchCount=result.operations.size();
  contract.attributes.clear(); for (const auto& a : result.attributes) contract.attributes.push_back(a.descriptor);
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

inline std::optional<ValueDescriptor> lowerFaceIndexField(const ValueDescriptor& source,
    StableId valueId, const PortContract& contract, std::string& error) {
  if (source.carrier!=CarrierKind::geometry3D || source.operations.size()>=kMaximumOperations
      || !source.spectrumFields.empty() || !source.scoreFields.empty()) {
    error="Face index requires static triangle geometry"; return std::nullopt;
  }
  ValueDescriptor result; result.carrier=CarrierKind::field; result.stableId=valueId;
  result.sourceStableId=source.sourceStableId; result.sourceRevision=source.sourceRevision;
  result.fieldDomain=Domain::face; result.fieldValueType=ValueType::integer; result.fieldInterpolation=Interpolation::constant;
  FieldData field; field.elements.resize(std::get<GeometryData>(source.data).indices.size()/3);
  for (std::size_t i=0;i<field.elements.size();++i) field.elements[i].components[0]=static_cast<double>(i);
  result.fieldDomainCardinality=field.elements.size(); result.data=std::move(field); result.operations=source.operations;
  FieldOperationSettings settings; settings.domain=Domain::face; settings.valueType=ValueType::integer;
  result.operations.push_back({valueId,OperationCode::fieldFaceIndex,source.stableId,valueId,{},0,{},settings});
  result.dispatchCount=result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

inline PortContract withAttributeContract(PortContract contract, const ValueDescriptor& value) {
  contract.attributes.clear();
  for (const auto& attribute : value.attributes) contract.attributes.push_back(attribute.descriptor);
  return contract;
}

inline std::optional<ValueDescriptor> lowerConstructedField(
    StableId valueId, OperationCode code, FieldOperationSettings settings,
    const ValueDescriptor* a, const ValueDescriptor* b, const ValueDescriptor* c,
    const PortContract& contract, std::string& error) {
  if (contract.carrier!=CarrierKind::field || contract.fieldDomain!=settings.domain
      || contract.fieldValueType!=settings.valueType) {
    error="Field output contract does not match its domain or type"; return std::nullopt;
  }
  ValueDescriptor result;
  result.carrier=CarrierKind::field; result.stableId=valueId;
  result.sourceStableId=a ? a->sourceStableId : valueId;
  result.sourceRevision=a ? a->sourceRevision : 1;
  result.fieldDomain=settings.domain; result.fieldValueType=settings.valueType;
  result.fieldInterpolation=Interpolation::constant;
  const auto types=fieldInputTypes(code,settings.valueType,settings.mode);
  const auto input=[&](const ValueDescriptor* value, ValueType type)->const FieldData* {
    if (!value) return nullptr;
    if (value->carrier!=CarrierKind::field || value->fieldDomain!=settings.domain || value->fieldValueType!=type
        || !std::holds_alternative<FieldData>(value->data)
        || !appendReplaySpectrumBindings(result.spectrumFields,*value,error)
        || !appendReplayScoreBindings(result.scoreFields,*value,error)
        || !appendUniqueOperations(result.operations,value->operations,error)) return nullptr;
    return &std::get<FieldData>(value->data);
  };
  const auto* af=input(a,types.first); const auto* bf=input(b,types.second);
  const auto* cf=input(c,code==OperationCode::fieldSelect ? ValueType::boolean : ValueType::floatValue);
  if ((a && !af) || (b && !bf) || (c && !cf)) {
    if (error.empty()) error="Constructed fields require supported inputs with matching domain and type";
    return std::nullopt;
  }
  settings.auxiliaryInputStableId=c ? c->stableId : 0;
  OperationRecord operation{valueId,code,a ? a->stableId : valueId,valueId,{},b ? b->stableId : 0,{},settings};
  FieldData field;
  if (result.operations.size()>=kMaximumOperations
      || !materializeConstructedField(operation,af,bf,cf,field,contract.maxFieldElements,error)) return std::nullopt;
  result.fieldDomainCardinality=field.elements.size(); result.data=std::move(field);
  result.operations.push_back(operation); result.dispatchCount=result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

inline std::optional<ValueDescriptor> lowerNamedAttribute(
    const ValueDescriptor& source, const ValueDescriptor* field, StableId valueId,
    OperationCode code, const std::string& name, ValueType type,
    PortContract contract, std::string& error, const AdmissionContext& context = {},
    float preserveLength = 0) {
  if (elementDomain(source.carrier)==Domain::unspecified || !hasOnlyOrderedSpectrumBindings(source)
      || !source.scoreFields.empty()) {
    error="Named attributes require static geometry, points, curves or instances"; return std::nullopt;
  }
  FieldOperationSettings settings;
  settings.domain=code==OperationCode::readNamedAttribute && source.carrier==CarrierKind::geometry3D
      && contract.fieldDomain==Domain::face ? Domain::face : elementDomain(source.carrier);
  if (source.carrier==CarrierKind::curves3D && ((code==OperationCode::readNamedAttribute && contract.fieldDomain==Domain::spline)
      || (code==OperationCode::storeNamedAttribute && field && field->fieldDomain==Domain::spline))) settings.domain=Domain::spline;
  settings.valueType=type; settings.attributeName=name;
  // Set Normal stores parameters[0] as its preserve-length flag: 0 normalizes (the
  // original behavior, so saved graphs keep their meaning) and 1 keeps authored length.
  const std::array<float,4> parameters{code==OperationCode::setNormal ? preserveLength : 0.0f,0,0,0};
  OperationRecord operation{valueId,code,source.stableId,valueId,parameters,field ? field->stableId : 0,{},settings};
  auto result=source; result.stableId=valueId;
  if (code==OperationCode::readNamedAttribute) {
    const auto* attribute=findNamedAttribute(source.attributes,name);
    if (!attribute || attribute->descriptor.domain!=settings.domain || attribute->descriptor.valueType!=type) {
      error="Named attribute is missing or has the wrong type"; return std::nullopt;
    }
    result.carrier=CarrierKind::field; result.data=FieldData{attribute->elements}; result.attributes.clear();
    result.fieldDomain=settings.domain; result.fieldValueType=type; result.fieldInterpolation=Interpolation::constant;
    result.fieldDomainCardinality=attribute->elements.size();
  } else {
    FieldData positions;
    if (!field || field->carrier!=CarrierKind::field || field->fieldDomain!=settings.domain
        || field->fieldValueType!=type || !std::holds_alternative<FieldData>(field->data)
        || !hasOnlyOrderedSpectrumBindings(*field) || !field->scoreFields.empty()
        || (code==OperationCode::setNormal && source.carrier!=CarrierKind::geometry3D)
        || !appendUniqueOperations(result.operations,field->operations,error)
        || !appendReplaySpectrumBindings(result.spectrumFields,*field,error)
        || !materializeElementField(source.data,source.carrier,OperationCode::positionField,positions,kMaximumFieldElements,error)) {
      if (error.empty()) error="Named attribute requires a matching field on its source domain";
      return std::nullopt;
    }
    if (settings.domain==Domain::spline) positions.elements.resize(std::get<CurvesData>(source.data).splines.size());
    if (!materializeStoredAttribute(result.attributes,positions.elements.size(),std::get<FieldData>(field->data),operation,error)) return std::nullopt;
  }
  if (result.operations.size()>=kMaximumOperations) { error="Named attribute exceeds the operation budget"; return std::nullopt; }
  result.operations.push_back(operation); result.dispatchCount=result.operations.size();
  if (!admitValue(result,withAttributeContract(contract,result),{},context,error)) return std::nullopt;
  return result;
}

inline std::optional<ValueDescriptor> lowerInstanceAppearance(
    const ValueDescriptor& source, const ValueDescriptor& field, const ValueDescriptor* mask,
    StableId valueId, const std::string& name, PortContract contract, std::string& error) {
  const auto type=name=="opacity" || isInstanceSurfaceScalar(name) ? ValueType::floatValue : ValueType::color;
  const auto matches=[&](const ValueDescriptor& value,ValueType expected) {
    return value.carrier==CarrierKind::field && value.fieldDomain==Domain::instance
        && value.fieldValueType==expected && std::holds_alternative<FieldData>(value.data)
        && hasOnlyOrderedSpectrumBindings(value) && hasOnlyOrderedScoreBindings(value);
  };
  if (source.carrier!=CarrierKind::instances3D || contract.carrier!=CarrierKind::instances3D
      || !hasOnlyOrderedScoreBindings(source) || !matches(field,type) || (mask && !matches(*mask,ValueType::boolean))) {
    error="Instance appearance requires Instances3D, a matching typed value field and an optional Boolean selection";
    return std::nullopt;
  }
  auto result=source; result.stableId=valueId;
  FieldOperationSettings settings;
  settings.domain=Domain::instance; settings.valueType=type; settings.attributeName=name;
  settings.auxiliaryInputStableId=mask ? mask->stableId : 0;
  OperationRecord operation{valueId,OperationCode::setInstanceAppearance,source.stableId,valueId,{},field.stableId,{},settings};
  if (!appendUniqueOperations(result.operations,field.operations,error)
      || !appendReplaySpectrumBindings(result.spectrumFields,field,error)
      || !appendReplayScoreBindings(result.scoreFields,field,error)
      || (mask && !appendReplayScoreBindings(result.scoreFields,*mask,error))
      || (mask && !appendReplaySpectrumBindings(result.spectrumFields,*mask,error))
      || (mask && !appendUniqueOperations(result.operations,mask->operations,error))
      || result.operations.size()>=kMaximumOperations
      || !materializeInstanceAppearance(result.attributes,std::get<InstancesData>(source.data).instances.size(),
          std::get<FieldData>(field.data),mask ? &std::get<FieldData>(mask->data) : nullptr,operation,error)) return std::nullopt;
  result.operations.push_back(operation); result.dispatchCount=result.operations.size();
  if (!admitValue(result,withAttributeContract(contract,result),error)) return std::nullopt;
  return result;
}

inline std::optional<ValueDescriptor> lowerMaterialIndices(const ValueDescriptor& source,
    const ValueDescriptor& field, StableId valueId, PortContract contract, std::string& error) {
  const auto domain=source.carrier==CarrierKind::geometry3D ? Domain::face : Domain::instance;
  if ((source.carrier!=CarrierKind::geometry3D && source.carrier!=CarrierKind::instances3D)
      || field.carrier!=CarrierKind::field || field.fieldDomain!=domain || field.fieldValueType!=ValueType::integer
      || !field.spectrumFields.empty() || !field.scoreFields.empty()) {
    error="Set Material Index requires a face or instance Integer field"; return std::nullopt;
  }
  auto result=source; result.stableId=valueId;
  FieldOperationSettings settings; settings.domain=domain; settings.valueType=ValueType::integer; settings.attributeName="materialIndex";
  OperationRecord op{valueId,OperationCode::setMaterialIndex,source.stableId,valueId,{},field.stableId,{},settings};
  const auto count=source.carrier==CarrierKind::geometry3D ? std::get<GeometryData>(source.data).indices.size()/3
      : std::get<InstancesData>(source.data).instances.size();
  if (!appendUniqueOperations(result.operations,field.operations,error) || result.operations.size()>=kMaximumOperations
      || !materializeStoredAttribute(result.attributes,count,std::get<FieldData>(field.data),op,error)) return std::nullopt;
  result.operations.push_back(op); result.dispatchCount=result.operations.size();
  if (!admitValue(result,withAttributeContract(contract,result),error)) return std::nullopt;
  return result;
}

inline std::optional<ValueDescriptor> lowerNearestField(
    const ValueDescriptor& source, const ValueDescriptor& target, StableId valueId,
    OperationCode code, FieldOperationSettings settings,
    const PortContract& contract, std::string& error) {
  if (target.carrier!=CarrierKind::geometry3D
      || !source.scoreFields.empty() || !target.scoreFields.empty()
      || contract.carrier!=CarrierKind::field || settings.domain!=elementDomain(source.carrier)
      || contract.fieldDomain!=settings.domain || contract.fieldValueType!=settings.valueType) {
    error="Nearest field requires typed source and target geometry with supported replay bindings"; return std::nullopt;
  }
  OperationRecord operation{valueId,code,source.stableId,valueId,{},target.stableId,{},settings};
  FieldData field;
  if (!materializeNearestField(source.data,source.carrier,std::get<GeometryData>(target.data),
                               target.attributes,operation,field,contract.maxFieldElements,error)) return std::nullopt;
  ValueDescriptor result;
  result.carrier=CarrierKind::field; result.stableId=valueId; result.sourceStableId=source.sourceStableId;
  result.sourceRevision=source.sourceRevision; result.fieldDomain=settings.domain;
  result.fieldValueType=settings.valueType; result.fieldInterpolation=Interpolation::constant;
  result.fieldDomainCardinality=field.elements.size(); result.data=std::move(field);
  result.operations=source.operations;
  if (!spectrum::appendOrderedBindings(result.spectrumFields,source.spectrumFields,error)
      || !spectrum::appendOrderedBindings(result.spectrumFields,target.spectrumFields,error)
      || !appendUniqueOperations(result.operations,target.operations,error) || result.operations.size()>=kMaximumOperations)
    return std::nullopt;
  result.operations.push_back(operation); result.dispatchCount=result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

// The source geometry supplies the named attribute. The Integer index field
// chooses one element per sampling position in the same domain.
inline std::optional<ValueDescriptor> lowerSampleIndexField(
    const ValueDescriptor& source, const ValueDescriptor& index, StableId valueId,
    FieldOperationSettings settings, const PortContract& contract, std::string& error) {
  if (source.carrier==CarrierKind::field || elementDomain(source.carrier)!=settings.domain
      || !source.spectrumFields.empty() || !source.scoreFields.empty()
      || index.carrier!=CarrierKind::field || index.fieldDomain!=settings.domain
      || index.fieldValueType!=ValueType::integer || !std::holds_alternative<FieldData>(index.data)
      || !index.spectrumFields.empty() || !index.scoreFields.empty()
      || contract.carrier!=CarrierKind::field || contract.fieldDomain!=settings.domain
      || contract.fieldValueType!=settings.valueType || settings.mode!=0 || settings.auxiliaryInputStableId!=0) {
    error="Sample index requires a typed source and an Integer index field in the same domain"; return std::nullopt;
  }
  OperationRecord operation{valueId,OperationCode::fieldSampleIndex,source.stableId,valueId,{},index.stableId,{},settings};
  FieldData field;
  const auto* attribute=findNamedAttribute(source.attributes,settings.attributeName);
  if (!materializeSampleIndexField(attribute,std::get<FieldData>(index.data),operation,field,
          contract.maxFieldElements,error)) return std::nullopt;
  ValueDescriptor result;
  result.carrier=CarrierKind::field; result.stableId=valueId; result.sourceStableId=source.sourceStableId;
  result.sourceRevision=source.sourceRevision; result.fieldDomain=settings.domain;
  result.fieldValueType=settings.valueType; result.fieldInterpolation=Interpolation::constant;
  result.fieldDomainCardinality=field.elements.size(); result.data=std::move(field);
  result.operations=source.operations;
  if (!appendUniqueOperations(result.operations,index.operations,error) || result.operations.size()>=kMaximumOperations)
    return std::nullopt;
  result.operations.push_back(operation); result.dispatchCount=result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

// Retain authored rest geometry and settings. Heights exist only during replay.
inline std::optional<ValueDescriptor> lowerWaterSurfaceSimulation(
    const ValueDescriptor& source, StableId valueId, StableId firstFieldId,
    double propagationSpeed, double damping, double scoreImpulseGain, double audioOnsetGain,
    const PortContract& contract, std::string& error, double surfaceMassPerArea = 0) {
  if (source.carrier != CarrierKind::geometry3D || contract.carrier != CarrierKind::geometry3D
      || !std::holds_alternative<GeometryData>(source.data) || !source.spectrumFields.empty()
      || !source.scoreFields.empty() || hasRuntimeFields(source) || source.operations.size() > kMaximumOperations - 3
      || valueId == source.stableId || !validStableId(firstFieldId) || !validStableId(firstFieldId + 1)) {
    error = "Water simulation requires one static retained XY rest grid"; return std::nullopt;
  }
  if (findNamedAttribute(source.attributes,"normal")) {
    error = "Water surface requires scene-derived normals; retained normal attributes are unavailable"; return std::nullopt;
  }
  FieldOperationSettings settings; settings.domain = Domain::vertex; settings.valueType = ValueType::floatValue;
  // Mass uses bodyMass units per simulation-space geometry-unit squared.
  // Zero preserves the authored uncoupled water contract.
  settings.values = {propagationSpeed, damping, scoreImpulseGain, audioOnsetGain,
      surfaceMassPerArea == 0 ? 0 : surfaceMassPerArea, 0, 0, 0};
  OperationRecord op{firstFieldId, OperationCode::fieldWaterHeight, source.stableId, firstFieldId, {}, 0, {}, settings};
  FieldData field;
  if (!materializeWaterHeightField(op, std::get<GeometryData>(source.data), field, contract.maxVertices, error))
    return std::nullopt;
  ValueDescriptor height;
  height.carrier = CarrierKind::field; height.stableId = firstFieldId; height.sourceStableId = source.sourceStableId;
  height.sourceRevision = source.sourceRevision; height.fieldDomain = Domain::vertex;
  height.fieldValueType = ValueType::floatValue; height.fieldInterpolation = Interpolation::constant;
  height.fieldDomainCardinality = field.elements.size(); height.data = std::move(field);
  height.operations = source.operations; height.operations.push_back(op); height.dispatchCount = height.operations.size();
  PortContract vector; vector.carrier = CarrierKind::field; vector.fieldDomain = Domain::vertex;
  vector.fieldValueType = ValueType::vector; vector.fieldInterpolation = Interpolation::constant;
  vector.maxFieldElements = contract.maxVertices;
  settings.valueType = ValueType::vector; settings.mode = 1; settings.values = {0, 0, 1, 0, 0, 0, 0, 0};
  const auto offset = lowerConstructedField(firstFieldId + 1, OperationCode::fieldComposeVector,
      settings, &height, nullptr, nullptr, vector, error);
  return offset ? lowerVectorDisplacement(source, *offset, valueId, 1, withAttributeContract(contract, source), error)
                : std::nullopt;
}

// Local XY is the wave plane and displacement is along local Z. The authored
// settings lower to existing fields; frame replay derives triangle normals.
inline std::optional<ValueDescriptor> lowerDirectionalSurfaceWave(
    const ValueDescriptor& source, StableId valueId, StableId firstFieldId,
    double amplitude, double wavelength, double directionDegrees, double speed,
    double phaseRadians, double loopPeriod, const PortContract& contract, std::string& error) {
  constexpr double tau=6.283185307179586476925286766559;
  const std::array<double,6> settings{amplitude,wavelength,directionDegrees,speed,phaseRadians,loopPeriod};
  if (source.carrier!=CarrierKind::geometry3D || contract.carrier!=CarrierKind::geometry3D
      || !std::all_of(settings.begin(),settings.end(),[](double v){return finite(v);})
      || amplitude<0 || amplitude>1000 || wavelength<0.001 || wavelength>1000
      || std::abs(directionDegrees)>360 || std::abs(speed)>1000 || std::abs(phaseRadians)>1000
      || loopPeriod<0 || loopPeriod>86400 || (loopPeriod>0 && loopPeriod<0.001)
      || !validStableId(valueId) || !validStableId(firstFieldId) || firstFieldId>kMaximumStableId-9
      || !source.spectrumFields.empty() || !source.scoreFields.empty()
      || findNamedAttribute(source.attributes,"normal")) {
    error="Directional Surface Wave requires bounded mesh settings and derived normals"; return std::nullopt;
  }
  const auto cycles=speed*loopPeriod/wavelength;
  if (loopPeriod>0 && std::abs(cycles-std::round(cycles))>64*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(cycles))) {
    error="Wave loop period must contain a whole number of cycles: speed times period divided by wavelength";
    return std::nullopt;
  }
  for (StableId id=firstFieldId; id<firstFieldId+10; ++id)
    if (id==valueId || id==source.stableId || std::any_of(source.operations.begin(),source.operations.end(),
        [&](const auto& op){return op.stableId==id || op.outputStableId==id;})) {
      error="Directional Surface Wave field identities overlap existing geometry"; return std::nullopt;
    }
  if (const auto* mesh=std::get_if<GeometryData>(&source.data)) {
    const std::uint64_t vertices=mesh->positions.size();
    if (vertices>std::numeric_limits<std::uint64_t>::max()/kDirectionalSurfaceWaveBytesPerVertex) {
      error="Directional Surface Wave size overflows: its mesh has "+std::to_string(vertices)+" vertices.";
      return std::nullopt;
    }
  }
  PortContract vector;
  vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex; vector.fieldValueType=ValueType::vector;
  vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=contract.maxVertices;
  auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
  auto positions=lowerElementField(source,firstFieldId,OperationCode::positionField,vector,error);
  if (!positions) return std::nullopt;
  FieldOperationSettings field;
  field.domain=Domain::vertex; field.valueType=ValueType::floatValue;
  const auto angle=directionDegrees*tau/360;
  field.values={std::cos(angle),std::sin(angle),0,0,0,0,0,0};
  auto coordinate=lowerConstructedField(firstFieldId+1,OperationCode::fieldVectorMeasure,field,&*positions,nullptr,nullptr,scalar,error);
  if (!coordinate) return std::nullopt;
  field.mode=2; field.values={tau/wavelength,0,0,0,0,0,0,0};
  auto spatial=lowerConstructedField(firstFieldId+2,OperationCode::fieldMath,field,&*coordinate,nullptr,nullptr,scalar,error);
  if (!spatial) return std::nullopt;
  field.mode=loopPeriod>0 ? 1 : 0; field.values={loopPeriod>0 ? loopPeriod : 1,0,0,0,1,0,0,0};
  auto time=lowerConstructedField(firstFieldId+3,OperationCode::fieldTimeline,field,&*coordinate,nullptr,nullptr,scalar,error);
  if (!time) return std::nullopt;
  field.mode=2; field.values={-tau*(loopPeriod>0 ? cycles : speed/wavelength),0,0,0,0,0,0,0};
  auto temporal=lowerConstructedField(firstFieldId+4,OperationCode::fieldMath,field,&*time,nullptr,nullptr,scalar,error);
  if (!temporal) return std::nullopt;
  field.mode=0; field.values={};
  auto combined=lowerConstructedField(firstFieldId+5,OperationCode::fieldMath,field,&*spatial,&*temporal,nullptr,scalar,error);
  if (!combined) return std::nullopt;
  field.values[0]=phaseRadians;
  auto phased=lowerConstructedField(firstFieldId+6,OperationCode::fieldMath,field,&*combined,nullptr,nullptr,scalar,error);
  if (!phased) return std::nullopt;
  field.mode=7; field.values={};
  auto sine=lowerConstructedField(firstFieldId+7,OperationCode::fieldMath,field,&*phased,nullptr,nullptr,scalar,error);
  if (!sine) return std::nullopt;
  field.mode=2; field.values[0]=amplitude;
  auto height=lowerConstructedField(firstFieldId+8,OperationCode::fieldMath,field,&*sine,nullptr,nullptr,scalar,error);
  if (!height) return std::nullopt;
  field.valueType=ValueType::vector; field.mode=1; field.values={0,0,1,0,0,0,0,0};
  auto offset=lowerConstructedField(firstFieldId+9,OperationCode::fieldComposeVector,field,&*height,nullptr,nullptr,vector,error);
  return offset ? lowerVectorDisplacement(source,*offset,valueId,1,withAttributeContract(contract,source),error) : std::nullopt;
}

// The same plane and Z displacement as the directional wave, with heights from
// windSpectrumWaves. The band reads the admitted grid: four cells up to its full
// extent. Frame replay evaluates the field and derives triangle normals.
inline std::optional<ValueDescriptor> lowerWindSpectrumSurfaceWave(
    const ValueDescriptor& source, StableId valueId, StableId firstFieldId,
    double amplitude, double windSpeed, double headingDegrees,
    const PortContract& contract, std::string& error, const WindSpectrumShape& shape = {},
    double smallestWave = 0) {
  if (source.carrier!=CarrierKind::geometry3D || contract.carrier!=CarrierKind::geometry3D
      || !finite(amplitude) || amplitude<0 || amplitude>1000
      || !validStableId(valueId) || !validStableId(firstFieldId) || firstFieldId>kMaximumStableId-4
      || !source.spectrumFields.empty() || !source.scoreFields.empty()
      || findNamedAttribute(source.attributes,"normal")) {
    error="Wind Spectrum requires bounded mesh settings and derived normals"; return std::nullopt;
  }
  if (!validWindSpectrumSettings(windSpeed,headingDegrees,error) || !validWindSpectrumShape(shape,error)) return std::nullopt;
  if (!finite(smallestWave) || smallestWave<0) {
    error="Wind Spectrum refused: smallest wave must be zero or a positive length"; return std::nullopt;
  }
  const auto* mesh=std::get_if<GeometryData>(&source.data);
  if (!mesh) { error="Wind Spectrum requires a regular XY grid"; return std::nullopt; }
  WaterSurfaceGrid grid;
  if (!validateWaterSurfaceGrid(*mesh,grid,error)) return std::nullopt;
  for (StableId id=firstFieldId; id<firstFieldId+5; ++id)
    if (id==valueId || id==source.stableId || std::any_of(source.operations.begin(),source.operations.end(),
        [&](const auto& op){return op.stableId==id || op.outputStableId==id;})) {
      error="Wind Spectrum field identities overlap existing geometry"; return std::nullopt;
    }
  PortContract vector;
  vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex; vector.fieldValueType=ValueType::vector;
  vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=contract.maxVertices;
  auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
  auto positions=lowerElementField(source,firstFieldId,OperationCode::positionField,vector,error);
  if (!positions) return std::nullopt;
  FieldOperationSettings field;
  field.domain=Domain::vertex; field.valueType=ValueType::floatValue;
  // Timeline Field needs a float input for its cardinality. A zero dot product gives one per vertex.
  auto zero=lowerConstructedField(firstFieldId+1,OperationCode::fieldVectorMeasure,field,&*positions,nullptr,nullptr,scalar,error);
  if (!zero) return std::nullopt;
  field.values={1,0,0,0,1,0,0,0};
  auto time=lowerConstructedField(firstFieldId+2,OperationCode::fieldTimeline,field,&*zero,nullptr,nullptr,scalar,error);
  if (!time) return std::nullopt;
  // Waves shorter than four grid cells alias, so Smallest Wave only raises that floor.
  const auto shortest=std::max(4*std::min(grid.spacingX,grid.spacingY),smallestWave);
  const auto extent=std::max((grid.columns-1)*grid.spacingX,(grid.rows-1)*grid.spacingY);
  field.mode=kNoiseFieldWindSpectrumMode;
  field.values={windSpeed,headingDegrees,windSpectrumRmsHeight(amplitude,windSpeed),shortest,std::max(shortest,extent),
      shape.alignment,shape.depth,shape.loopPeriod};
  auto height=lowerConstructedField(firstFieldId+3,OperationCode::fieldNoise,field,&*positions,&*time,nullptr,scalar,error);
  if (!height) return std::nullopt;
  field.valueType=ValueType::vector; field.mode=1; field.values={0,0,1,0,0,0,0,0};
  auto offset=lowerConstructedField(firstFieldId+4,OperationCode::fieldComposeVector,field,&*height,nullptr,nullptr,vector,error);
  return offset ? lowerVectorDisplacement(source,*offset,valueId,1,withAttributeContract(contract,source),error) : std::nullopt;
}
// The Ocean node. Local XY is the sea plane; heights come from the ocean spectrum
// and choppiness scales the horizontal Gerstner motion. Zero for the smallest wave
// or the size follows the grid: four cells up to its full extent, the same band
// the wind spectrum reads. Foam from the displaced surface is stored as the vertex
// Float attribute "foam" before the displacement, so Read Attribute nodes see it
// on the displaced mesh. Frame replay derives triangle normals.
//
// A detail falloff above 0 makes wave detail follow wavelength: each vertex's
// footprint is the larger of its own grid cell and falloff times its distance
// from the focus point (local XY), and waves shorter than two footprints fade
// out, so a coarse or distant region keeps no wave it would alias. A falloff of
// one pixel's angle in radians with the focus under the camera filters by screen
// size.
inline constexpr std::size_t kOceanFieldIdentities = 9;
inline constexpr double kOceanMaximumDetailFalloff = 1;
inline std::optional<ValueDescriptor> lowerOceanSurface(
    const ValueDescriptor& source, StableId valueId, StableId firstFieldId,
    OceanSettings ocean, double choppiness, double foamCoverage,
    const PortContract& contract, std::string& error,
    double detailFalloff = 0, double focusX = 0, double focusY = 0) {
  if (source.carrier!=CarrierKind::geometry3D || contract.carrier!=CarrierKind::geometry3D
      || !validStableId(valueId) || !validStableId(firstFieldId)
      || firstFieldId>kMaximumStableId-(kOceanFieldIdentities-1)
      || !source.spectrumFields.empty() || !source.scoreFields.empty()
      || findNamedAttribute(source.attributes,"normal")) {
    error="Ocean requires a static XY grid with derived normals"; return std::nullopt;
  }
  if (!(choppiness>=0 && choppiness<=kOceanMaximumChoppiness)) {
    error="Ocean refused: choppiness "+std::to_string(choppiness)+" is outside 0 to "
        +std::to_string(kOceanMaximumChoppiness);
    return std::nullopt;
  }
  if (!(foamCoverage>=0 && foamCoverage<=kOceanMaximumFoamCoverage)) {
    error="Ocean refused: foam coverage "+std::to_string(foamCoverage)+" is outside 0 to "
        +std::to_string(kOceanMaximumFoamCoverage);
    return std::nullopt;
  }
  if (!(detailFalloff>=0 && detailFalloff<=kOceanMaximumDetailFalloff) || !std::isfinite(focusX) || !std::isfinite(focusY)) {
    error="Ocean refused: detail falloff "+std::to_string(detailFalloff)+" is outside 0 to "
        +std::to_string(kOceanMaximumDetailFalloff)+" or its focus is not finite";
    return std::nullopt;
  }
  const auto* mesh=std::get_if<GeometryData>(&source.data);
  if (!mesh) { error="Ocean requires an XY grid"; return std::nullopt; }
  WaterSurfaceGrid grid;
  if (!validateWaterSurfaceGrid(*mesh,grid,error,true)) return std::nullopt;
  if (ocean.smallestWave==0) ocean.smallestWave=4*std::min(grid.spacingX,grid.spacingY);
  if (ocean.size==0)
    ocean.size=std::max(ocean.smallestWave*2,std::max((grid.columns-1)*grid.spacingX,(grid.rows-1)*grid.spacingY));
  if (!validOceanSettings(ocean,error)) return std::nullopt;
  for (StableId id=firstFieldId; id<firstFieldId+kOceanFieldIdentities; ++id)
    if (id==valueId || id==source.stableId || std::any_of(source.operations.begin(),source.operations.end(),
        [&](const auto& op){return op.stableId==id || op.outputStableId==id;})) {
      error="Ocean field identities overlap existing geometry"; return std::nullopt;
    }
  PortContract vector;
  vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex; vector.fieldValueType=ValueType::vector;
  vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=contract.maxVertices;
  auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
  auto positions=lowerElementField(source,firstFieldId,OperationCode::positionField,vector,error);
  if (!positions) return std::nullopt;
  FieldOperationSettings field;
  field.domain=Domain::vertex; field.valueType=ValueType::floatValue;
  // Timeline Field needs a float input for its cardinality. A zero dot product gives one per vertex.
  auto zero=lowerConstructedField(firstFieldId+1,OperationCode::fieldVectorMeasure,field,&*positions,nullptr,nullptr,scalar,error);
  if (!zero) return std::nullopt;
  // A looping ocean reads the loop phase, so its waves repeat exactly at any timeline time.
  field.mode=ocean.loopPeriod>0 ? 1 : 0;
  field.values={ocean.loopPeriod>0 ? ocean.loopPeriod : 1,0,0,0,1,0,0,0};
  auto time=lowerConstructedField(firstFieldId+2,OperationCode::fieldTimeline,field,&*zero,nullptr,nullptr,scalar,error);
  if (!time) return std::nullopt;
  std::optional<ValueDescriptor> footprint;
  if (detailFalloff>0) {
    field.mode=2; field.values={focusX,focusY,0,0,0,0,0,0};
    auto distance=lowerConstructedField(firstFieldId+7,OperationCode::fieldVectorMeasure,field,&*positions,nullptr,nullptr,scalar,error);
    if (!distance) return std::nullopt;
    field.mode=2; field.values={detailFalloff,0,0,0,0,0,0,0};
    footprint=lowerConstructedField(firstFieldId+8,OperationCode::fieldMath,field,&*distance,nullptr,nullptr,scalar,error);
    if (!footprint) return std::nullopt;
  }
  field.valueType=ValueType::vector; field.mode=kNoiseFieldOceanMode; field.values=oceanFieldValues(ocean);
  auto raw=lowerConstructedField(firstFieldId+3,OperationCode::fieldNoise,field,&*positions,&*time,
      footprint ? &*footprint : nullptr,vector,error);
  if (!raw) return std::nullopt;
  field.mode=2; field.values={choppiness,choppiness,1,0,0,0,0,0};
  auto offset=lowerConstructedField(firstFieldId+4,OperationCode::fieldVectorMath,field,&*raw,nullptr,nullptr,vector,error);
  if (!offset) return std::nullopt;
  field.valueType=ValueType::floatValue; field.mode=kNoiseFieldOceanFoamMode; field.values={foamCoverage,0,0,0,0,0,0,0};
  auto foam=lowerConstructedField(firstFieldId+5,OperationCode::fieldNoise,field,&*positions,&*offset,nullptr,scalar,error);
  if (!foam) return std::nullopt;
  auto withFoam=lowerNamedAttribute(source,&*foam,firstFieldId+6,OperationCode::storeNamedAttribute,
      "foam",ValueType::floatValue,contract,error);
  if (!withFoam) return std::nullopt;
  return lowerVectorDisplacement(*withFoam,*offset,valueId,1,withAttributeContract(contract,*withFoam),error);
}
} // namespace videowire::geometry
