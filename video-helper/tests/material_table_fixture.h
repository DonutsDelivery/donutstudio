#pragma once

#include "instance_appearance_fixture.h"
#include "../../shared/GeometryMaterialTable.h"

inline std::optional<videowire::geometry::ValueDescriptor> materialTableFixture(bool instances,
    bool appearance, std::string& error) {
  using namespace videowire::geometry;
  const auto contract=instances ? appearanceInstanceContract() : constructedFieldMeshContract();
  auto value=instances ? instanceAppearanceFixture(appearance ? 4 : 0,error)
      : lowerGrid(9400,9400,1,2,2,2,contract,error);
  auto indices=value ? (instances ? lowerElementField(*value,9401,OperationCode::indexField,
      constructedFieldContract(Domain::instance,ValueType::integer),error)
      : lowerFaceIndexField(*value,9401,constructedFieldContract(Domain::face,ValueType::integer),error)) : std::nullopt;
  value=value && indices ? lowerMaterialIndices(*value,*indices,9402,contract,error) : std::nullopt;
  const std::array<double,8> red{1,0.05,0.05,0.25,0,0,0,0.7},green{0.05,1,0.05,0,0.25,0,0,0.7};
  for (std::uint8_t slot=0;value && slot<(instances ? 4 : 2);++slot)
    value=lowerMaterialTableSlot(*value,9410+slot,9420+slot,slot,slot%2 ? green : red,1,contract,error);
  return value;
}

template <typename ReadPixels>
bool verifyNativeMaterialTables(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  std::vector<std::uint8_t> plainInstances;
  for (unsigned variant=0;variant<3;++variant) {
    const bool instances=variant>0;
    const auto value=materialTableFixture(instances,variant==2,error);
    if (!value) return false;
    const auto contract=withAttributeContract(instances ? appearanceInstanceContract() : constructedFieldMeshContract(),*value);
    const auto admitted=admitValue(*value,contract,error);
    if (!admitted) return false;
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    const auto preview=runtime.admitPreview({1,1,1,9413,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,9413,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    const auto draw=executeNativeGeometry(backend,*preview,96,96,error);
    const auto exportDraw=executeNativeGeometry(backend,*exported,96,96,error);
    if (!draw || !exportDraw) return false;
    const auto pixels=readPixels(draw->frame);
    if (pixels.empty() || pixels!=readPixels(exportDraw->frame)
        || draw->stats.vertexBytes!=4*sizeof(HarmonicMIDI::grid::SceneVertex)
        || draw->stats.ordinaryDrawCount!=(instances ? 4 : 2)
        || (instances && draw->drawnInstanceIds!=std::vector<StableId>{1,2,3,4})) {
      error="Material batches must share vertices, preserve instance identities and match preview with export"; return false;
    }
    if (variant==1) plainInstances=pixels;
    if (variant==2 && pixels==plainInstances) { error="Instance appearance must affect table-selected materials"; return false; }
    if (variant<2) {
      std::size_t green=0,red=0;
      for (std::size_t i=0;i<pixels.size();i+=4) {
        const auto rb=std::max(pixels[i],pixels[i+2]);
        if (pixels[i+1]>20 && pixels[i+1]>2*rb) ++green;
        if (rb>20 && rb>2*pixels[i+1]) ++red;
      }
      if (red<10 || green<10) { error="Material indices must produce visible red and green face or instance batches"; return false; }
    }
  }
  return true;
}
