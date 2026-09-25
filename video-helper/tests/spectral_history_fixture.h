#pragma once

#include "../src/geometry_core_backend.h"

inline std::optional<videowire::geometry::ValueDescriptor> spectralHistoryTerrain(
    videowire::geometry::PortContract& contract, std::string& error) {
  using namespace videowire::geometry;
  contract.carrier=CarrierKind::geometry3D; contract.overflow=OverflowPolicy::reject;
  contract.maxVertices=16; contract.maxIndices=54;
  auto value=lowerGrid(8701,8701,1,4,4,0.5f,contract,error);
  if (!value) return std::nullopt;
  spectrum::Binding binding;
  binding.fieldStableId=8702; binding.sourceGeometryStableId=8701;
  binding.consumerGeometryStableId=8703; binding.firstBand=8; binding.lastBand=8;
  binding.direction={0,0.6f,1}; binding.history.enabled=true;
  binding.history.depth=4; binding.history.spacingSeconds=0.25f;
  for (std::size_t index=0;index<16;++index) {
    binding.coordinates.push_back(static_cast<float>(index%4)/3);
    binding.history.coordinates.push_back(static_cast<float>(index/4)/3);
  }
  value->operations.push_back({8702,OperationCode::evaluateField,8702,8702,{}});
  value->operations.push_back({8703,OperationCode::transform,8701,8703,{}});
  value->stableId=8703; value->dispatchCount=3; value->spectrumFields={binding};
  return value;
}

inline videowire::geometry::spectrum::Bands spectralHistoryPulse(double seconds) {
  videowire::geometry::spectrum::Bands bands {};
  bands[8]=seconds>=0.4 && seconds<0.6 ? 0.8f : 0.0f;
  return bands;
}

template <typename ReadPixels>
bool verifyNativeSpectralHistory(arbitgpu::NativeFixtureSceneBackend& backend,
                                 ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  PortContract contract;
  const auto value=spectralHistoryTerrain(contract,error);
  const auto admitted=value ? admitValue(*value,contract,error) : std::nullopt;
  if (!admitted) return false;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  GeometryCorePlanRuntime runtime(capabilities);
  const auto bytes=lowerRuntimePlan(contract,*admitted);
  auto preview=runtime.admitPreview({1,1,1,8703,1,PlanUse::preview},bytes,{}, {},error);
  auto exported=runtime.admitExport({1,1,1,8703,1,PlanUse::exportRender},bytes,{}, {},error);
  if (!preview || !exported) return false;
  SpectrumEvaluation input;
  input.historyFeaturesAt=spectralHistoryPulse;
  input.timeSeconds=0.2; input.bands=spectralHistoryPulse(0.2);
  auto before=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
  input.timeSeconds=1.0; input.bands=spectralHistoryPulse(1.0);
  auto history=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
  SpectrumEvaluation exportInput;
  exportInput.timeSeconds=1.0; exportInput.historyFeaturesAt=spectralHistoryPulse;
  auto rendered=executeNativeGeometry(backend,*exported,64,64,error,false,{},&exportInput);
  if (!before || !history || !rendered) return false;
  const auto beforePixels=readPixels(before->frame), historyPixels=readPixels(history->frame);
  if (beforePixels.empty() || historyPixels==beforePixels || historyPixels!=readPixels(rendered->frame)) {
    error="Spectral History must show past audio while current bands are zero and match random-access export pixels";
    return false;
  }
  input.timeSeconds=0.2;
  auto sought=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
  input.timeSeconds=1.0;
  auto looped=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
  if (!sought || !looped || readPixels(sought->frame)!=beforePixels || readPixels(looped->frame)!=historyPixels) {
    error="Spectral History seek and loop did not restore the same native frame"; return false;
  }
  input.historyFeaturesAt={};
  if (executeNativeGeometry(backend,*preview,64,64,error,false,{},&input)
      || error.find("baked audio analysis")==std::string::npos) {
    error="Missing baked Spectral History must diagnose instead of drawing a current-spectrum substitute";
    return false;
  }
  error.clear(); return true;
}
