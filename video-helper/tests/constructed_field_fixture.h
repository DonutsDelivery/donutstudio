#pragma once

#include "../src/geometry_core_backend.h"

inline videowire::geometry::PortContract constructedFieldMeshContract() {
  using namespace videowire::geometry;
  PortContract contract;
  contract.carrier=CarrierKind::geometry3D; contract.maxVertices=1024; contract.maxIndices=6144;
  contract.maxAttributes=32; return contract;
}

inline videowire::geometry::PortContract constructedFieldContract(
    videowire::geometry::Domain domain, videowire::geometry::ValueType type) {
  using namespace videowire::geometry;
  PortContract contract;
  contract.carrier=CarrierKind::field; contract.fieldDomain=domain; contract.fieldValueType=type;
  contract.fieldInterpolation=Interpolation::constant; contract.maxFieldElements=1024;
  contract.maxAttributes=32; return contract;
}

inline std::optional<videowire::geometry::ValueDescriptor> constructedFieldFixture(
    int variant, std::string& error) {
  using namespace videowire::geometry;
  const auto meshContract=constructedFieldMeshContract();
  auto source=lowerGrid(8101,8101,1,8,8,0.35f,meshContract,error);
  if (!source || variant==0) return source;
  auto target=lowerMeshGenerator(8102,8102,1,OperationCode::plane,{1.5f,1.5f,0,0},meshContract,error);
  Transform transform; transform.translation={0.4f,0.2f,0.3f};
  auto moved=target ? lowerTransformGeometry(*target,8103,transform,meshContract,error) : std::nullopt;
  FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::vector;
  const auto vectorContract=constructedFieldContract(Domain::vertex,ValueType::vector);
  if (variant==4) settings.values={0,0,1,2,0,0,0,0};
  auto offset=moved ? lowerNearestField(*source,*moved,8104,variant==3 ? OperationCode::fieldSurfaceOffset
      : variant==4 ? OperationCode::fieldRaycastPosition : OperationCode::fieldNearestDirection,settings,vectorContract,error) : std::nullopt;
  if (variant==4 && offset) {
    const auto position=lowerElementField(*source,8111,OperationCode::positionField,vectorContract,error);
    settings.values={}; settings.mode=1;
    offset=position ? lowerConstructedField(8112,OperationCode::fieldVectorMath,settings,&*offset,&*position,nullptr,vectorContract,error) : std::nullopt;
  }
  settings.mode=2; settings.values={0.6,0.6,1,0,0,0,0,0};
  auto scaled=offset ? lowerConstructedField(8105,OperationCode::fieldVectorMath,settings,&*offset,nullptr,nullptr,vectorContract,error) : std::nullopt;
  auto stored=scaled ? lowerNamedAttribute(*source,&*scaled,8106,OperationCode::storeNamedAttribute,"offset",ValueType::vector,meshContract,error) : std::nullopt;
  auto read=stored ? lowerNamedAttribute(*stored,nullptr,8107,OperationCode::readNamedAttribute,"offset",ValueType::vector,vectorContract,error) : std::nullopt;
  auto displaced=read ? lowerVectorDisplacement(*stored,*read,8108,1,withAttributeContract(meshContract,*stored),error) : std::nullopt;
  if (!displaced) return std::nullopt;
  settings.mode=0; settings.values=variant==1 ? std::array<double,8>{0,0,1,0,1,0,0,0} : std::array<double,8>{1,0,0,0,1,0,0,0};
  auto normal=lowerConstructedField(8109,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,vectorContract,error);
  return normal ? lowerNamedAttribute(*displaced,&*normal,8110,OperationCode::setNormal,"normal",ValueType::vector,meshContract,error) : std::nullopt;
}

template <typename ReadPixels>
bool verifyNativeConstructedFields(arbitgpu::NativeFixtureSceneBackend& backend,
                                    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  std::vector<std::uint8_t> previous;
  for (int variant=0;variant<5;++variant) {
    auto value=constructedFieldFixture(variant,error);
    if (!value) return false;
    const auto contract=withAttributeContract(constructedFieldMeshContract(),*value);
    const auto admitted=admitValue(*value,contract,error);
    if (!admitted) return false;
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    GeometryCorePlanRuntime runtime(capabilities);
    const auto preview=runtime.admitPreview({1,1,1,8110,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,8110,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    const auto draw=executeNativeGeometry(backend,*preview,64,64,error);
    const auto exportDraw=executeNativeGeometry(backend,*exported,64,64,error);
    if (!draw || !exportDraw) return false;
    const auto pixels=readPixels(draw->frame);
    if (pixels.empty() || pixels!=readPixels(exportDraw->frame) || draw->stats.ordinaryDrawCount!=1
        || (variant!=0 && pixels==previous)) {
      error="Field displacement and authored normals must change native pixels and agree between preview and export"; return false;
    }
    previous=pixels;
  }
  error.clear(); return true;
}
