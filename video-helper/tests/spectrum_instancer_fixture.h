#pragma once

#include "../src/geometry_core_backend.h"

template <typename ReadPixels>
bool verifyNativeSpectrumInstancer(arbitgpu::NativeFixtureSceneBackend& backend,
                                   ReadPixels readPixels, std::string& error,
                                   const std::string& surfaceMaterial = {}) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  PortContract meshContract;
  meshContract.carrier=CarrierKind::geometry3D; meshContract.overflow=OverflowPolicy::reject;
  meshContract.maxVertices=4; meshContract.maxIndices=6;
  PortContract contract;
  contract.carrier=CarrierKind::instances3D; contract.overflow=OverflowPolicy::reject;
  contract.maxInstances=64;
  const auto mesh=lowerGrid(921,921,1,2,2,1,meshContract,error);
  if (!mesh) return false;
  for (int arrangement=0;arrangement<4;++arrangement) {
    spectrum::Binding binding;
    binding.firstBand=8; binding.lastBand=arrangement==3 ? 8 : 11;
    binding.baseScale={0.35f,0.35f,0.35f}; binding.scaleResponse={0,3,0};
    binding.direction={0,0.6f,0}; binding.attackSeconds=0.1f; binding.releaseSeconds=0.3f;
    auto value=lowerSpectrumInstancer(*mesh,922,binding,
        {static_cast<float>(arrangement%3),2,2,2},contract,error);
    auto admitted=value ? admitValue(*value,contract,error) : std::nullopt;
    if (!admitted) return false;
    const auto bytes=lowerRuntimePlan(contract,*admitted,surfaceMaterial);
    GeometryCorePlanRuntime runtime(capabilities);
    auto preview=runtime.admitPreview({1,1,1,922,1,PlanUse::preview},bytes,{}, {},error);
    auto exported=runtime.admitExport({1,1,1,922,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    SpectrumEvaluation input;
    input.featuresAt=[](double seconds) {
      spectrum::Bands bands {}; bands[8]=seconds>=0.5 ? 0.8f : 0; return bands;
    };
    input.timeSeconds=0.2; input.bands=input.featuresAt(input.timeSeconds);
    auto silent=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    input.timeSeconds=0.8; input.bands=input.featuresAt(input.timeSeconds);
    auto active=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    auto exportInput=input; exportInput.followers.clear();
    auto exportDraw=executeNativeGeometry(backend,*exported,64,64,error,false,{},&exportInput);
    if (!silent || !active || !exportDraw) return false;
    const auto quietPixels=readPixels(silent->frame), activePixels=readPixels(active->frame);
    if (quietPixels.empty() || activePixels==quietPixels || activePixels!=readPixels(exportDraw->frame)
        || active->drawnInstanceIds!=silent->drawnInstanceIds
        || active->drawnInstanceIds!=exportDraw->drawnInstanceIds
        || active->drawnInstanceIds.size()!=(arrangement==3 ? 1u : 4u)
        || active->drawnInstanceIds.front()!=9
        || active->stats.vertexBytes!=4*sizeof(HarmonicMIDI::grid::SceneVertex)
        || active->stats.indexBytes!=6*sizeof(std::uint32_t)
        || (arrangement!=3 && (surfaceMaterial.empty()
            ? active->stats.submittedInstanceCount!=4 || active->stats.instancedDrawCount!=1
            : active->stats.ordinaryDrawCount!=4 || active->stats.instancedDrawCount!=0))) {
      error="Spectrum Instancer must change native pixels with stable IDs, shared source buffers and preview/export equality";
      return false;
    }
    input.timeSeconds=0.2; input.bands=input.featuresAt(input.timeSeconds);
    auto sought=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    if (!sought || readPixels(sought->frame)!=quietPixels) {
      error="Spectrum Instancer seek did not restore the earlier rendered frame";
      return false;
    }
  }
  error.clear();
  return true;
}
