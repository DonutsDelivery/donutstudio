#pragma once

#include "constructed_field_fixture.h"

inline videowire::geometry::PortContract appearanceInstanceContract() {
  videowire::geometry::PortContract contract;
  contract.carrier=videowire::geometry::CarrierKind::instances3D;
  contract.maxInstances=64; contract.maxAttributes=32;
  return contract;
}

inline std::optional<videowire::geometry::ValueDescriptor> instanceAppearanceFixture(
    unsigned variant, std::string& error) {
  using namespace videowire::geometry;
  const auto contract=appearanceInstanceContract();
  auto source=lowerGrid(9101,9101,1,2,2,1.2f,constructedFieldMeshContract(),error);
  auto prototype=lowerMeshGenerator(9102,9102,1,OperationCode::plane,{0.7f,0.7f,0,0},constructedFieldMeshContract(),error);
  PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=4;
  auto points=source ? lowerPointsFromVertices(*source,9103,pointsContract,error) : std::nullopt;
  auto instances=points && prototype ? lowerInstanceOnPoints(*points,prototype->stableId,9104,contract,error) : std::nullopt;
  if (!instances || !appendUniqueOperations(instances->operations,prototype->operations,error)) return std::nullopt;
  // The source must precede its instance consumer in the retained operation graph.
  std::rotate(instances->operations.begin(),instances->operations.end()-1,instances->operations.end());
  instances->dispatchCount=instances->operations.size();
  if (variant==0) return instances;
  const auto colorContract=constructedFieldContract(Domain::instance,ValueType::color);
  const auto floatContract=constructedFieldContract(Domain::instance,ValueType::floatValue);
  FieldOperationSettings settings; settings.domain=Domain::instance; settings.valueType=ValueType::color;
  settings.values={1,0,0,1,1,0,0,0};
  auto red=lowerConstructedField(9110,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,colorContract,error);
  settings.values={0,1,0,1,1,0,0,0};
  auto green=lowerConstructedField(9111,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,colorContract,error);
  auto index=lowerElementField(*instances,9112,OperationCode::indexField,
      constructedFieldContract(Domain::instance,ValueType::integer),error);
  settings.valueType=ValueType::floatValue; settings.values={};
  auto indexFloat=index ? lowerConstructedField(9113,OperationCode::fieldIntegerToFloat,settings,
      &*index,nullptr,nullptr,floatContract,error) : std::nullopt;
  settings.valueType=ValueType::boolean; settings.values[0]=2;
  auto selected=indexFloat ? lowerConstructedField(9114,OperationCode::fieldCompare,settings,
      &*indexFloat,nullptr,nullptr,constructedFieldContract(Domain::instance,ValueType::boolean),error) : std::nullopt;
  if (!red || !green || !selected) return std::nullopt;
  auto colored=lowerInstanceAppearance(*instances,*red,nullptr,9115,"color",contract,error);
  colored=colored ? lowerInstanceAppearance(*colored,*green,&*selected,9116,"color",contract,error) : std::nullopt;
  if (!colored || variant==1) return colored;
  auto glowing=lowerInstanceAppearance(*colored,*red,&*selected,9117,"emission",contract,error);
  if (!glowing || variant==2) return glowing;
  settings.valueType=ValueType::floatValue; settings.values={variant==3 ? 0.0 : 0.35,0,0,0,1,0,0,0};
  auto opacity=lowerConstructedField(9118,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,floatContract,error);
  return opacity ? lowerInstanceAppearance(*glowing,*opacity,&*selected,9119,"opacity",contract,error) : std::nullopt;
}

template <typename ReadPixels>
bool verifyNativeInstanceAppearance(arbitgpu::NativeFixtureSceneBackend& backend,
                                     ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  std::vector<std::uint8_t> previous,identityPixels;
  for (unsigned variant=0;variant<5;++variant) {
    const auto value=instanceAppearanceFixture(variant,error);
    if (!value) return false;
    const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
    const auto admitted=admitValue(*value,contract,error);
    if (!admitted) return false;
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    GeometryCorePlanRuntime runtime(capabilities);
    const auto preview=runtime.admitPreview({1,1,1,9119,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,9119,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    const auto draw=executeNativeGeometry(backend,*preview,64,64,error);
    const auto exportDraw=executeNativeGeometry(backend,*exported,64,64,error);
    if (!draw || !exportDraw) return false;
    const auto pixels=readPixels(draw->frame);
    if (pixels.empty() || pixels!=readPixels(exportDraw->frame) || (variant && pixels==previous)
        || draw->drawnInstanceIds!=std::vector<StableId>{1,2,3,4}
        || draw->drawnInstanceIds!=exportDraw->drawnInstanceIds
        || (variant<3 && draw->stats.instancedDrawCount!=1)
        || (variant>=3 && draw->stats.ordinaryDrawCount!=4)
        || draw->stats.vertexBytes!=4*sizeof(HarmonicMIDI::grid::SceneVertex)) {
      error="Instance appearance must change native pixels, preserve identities and shared mesh storage, and agree in preview/export";
      return false;
    }
    previous=pixels;
    if (variant==1) {
      std::size_t green=0,red=0;
      for (std::size_t i=0;i<pixels.size();i+=4) {
        const auto rb=std::max(pixels[i],pixels[i+2]);
        if (pixels[i+1]>20 && pixels[i+1]>2*rb) ++green;
        if (rb>20 && rb>2*pixels[i+1]) ++red;
      }
      if (green<10 || red<10) { error="A masked instance color field must produce separate red and green objects"; return false; }
    }
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.imageOutput=renderpassoutput::Output::ObjectId;
    const auto ids=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
    if (!ids) return false;
    const auto idPixels=readPixels(ids->frame);
    if (variant==0) identityPixels=idPixels;
    else if ((variant<3 && idPixels!=identityPixels) || (variant==3 && idPixels==identityPixels)) {
      error="Instance ID output must ignore color/emission and discard fully transparent instances"; return false;
    }
    if (variant==0) {
      std::set<std::array<std::uint8_t,3>> colors;
      for (std::size_t i=0;i<idPixels.size();i+=4)
        if (idPixels[i+3]) colors.insert({idPixels[i],idPixels[i+1],idPixels[i+2]});
      if (colors.size()<4) { error="Instanced ID output collapsed independent object identities"; return false; }
    }
    if (variant==2) {
      inputs.imageOutput=renderpassoutput::Output::Emission;
      const auto emission=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
      const auto exportEmission=executeNativeGeometry(backend,*exported,64,64,error,false,inputs);
      if (!emission || !exportEmission) return false;
      const auto emissionPixels=readPixels(emission->frame);
      std::size_t lit=0,dark=0;
      for (std::size_t i=0;i<emissionPixels.size();i+=4) {
        if (emissionPixels[i+3]==0) continue;
        const auto maximum=std::max({emissionPixels[i],emissionPixels[i+1],emissionPixels[i+2]});
        if (maximum>100) ++lit;
        if (maximum<5) ++dark;
      }
      if (emissionPixels!=readPixels(exportEmission->frame) || lit<10 || dark<10) {
        error="Emission output must retain selected emissive instances and agree in preview/export"; return false;
      }
    }
  }
  return true;
}
