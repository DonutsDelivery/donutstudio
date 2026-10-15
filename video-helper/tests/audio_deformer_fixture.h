#pragma once

#include "../src/geometry_core_backend.h"

template <typename ReadPixels>
bool verifyNativeAudioDeformer(arbitgpu::NativeFixtureSceneBackend& backend,
                               ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  PortContract contract;
  contract.carrier=CarrierKind::geometry3D; contract.overflow=OverflowPolicy::reject;
  contract.maxVertices=100; contract.maxIndices=600;
  const auto source=lowerGrid(941,941,1,9,9,0.25f,contract,error);
  if (!source) return false;
  const spectrum::FeaturesAt features=[](double seconds) {
    spectrum::Bands bands {}; bands.fill(seconds>=0.5 && seconds<1.5 ? 0.8f : 0); return bands;
  };
  for (int mode=0;mode<=10;++mode) {
    spectrum::Binding binding;
    binding.firstBand=0; binding.lastBand=23; binding.direction={0.5f,0.3f,1};
    binding.deformer.mode=static_cast<spectrum::Deformation>(mode);
    binding.deformer.frequency=3.7f; binding.deformer.phase=0.3f;
    binding.deformer.falloffRadius=3; binding.deformer.falloffPower=1.2f;
    binding.deformer.delaySeconds=0.1f; binding.deformer.threshold=0.2f;
    binding.deformer.hysteresis=0.1f; binding.deformer.peakHoldSeconds=0.1f;
    binding.deformer.quantizeSteps=16; binding.deformer.envelopePower=0.8f;
    binding.attackSeconds=0.05f; binding.releaseSeconds=0.1f;
    auto value=lowerAudioDeformer(*source,942,binding,contract,error);
    auto admitted=value ? admitValue(*value,contract,error) : std::nullopt;
    if (!admitted) return false;
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    GeometryCorePlanRuntime runtime(capabilities);
    auto preview=runtime.admitPreview({1,1,1,942,1,PlanUse::preview},bytes,{}, {},error);
    auto exported=runtime.admitExport({1,1,1,942,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    SpectrumEvaluation input; input.featuresAt=features;
    input.timeSeconds=0.2; input.bands=features(input.timeSeconds);
    auto silent=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    if (!silent) return false;
    const auto quietPixels=readPixels(silent->frame);
    input.timeSeconds=1.1; input.bands=features(input.timeSeconds);
    auto active=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    if (!active) return false;
    const auto activePixels=readPixels(active->frame);
    auto exportInput=input; exportInput.followers.clear();
    auto exportDraw=executeNativeGeometry(backend,*exported,64,64,error,false,{},&exportInput);
    if (!exportDraw) return false;
    if (quietPixels.empty() || activePixels==quietPixels || activePixels!=readPixels(exportDraw->frame)
        || active->stats.vertexBytes!=81*sizeof(HarmonicMIDI::grid::SceneVertex)
        || active->stats.indexBytes!=384*sizeof(std::uint32_t)) {
      error="Audio Deformer mode " + std::to_string(mode)
          + " must change native pixels with fixed topology and preview/export equality";
      return false;
    }
    input.timeSeconds=0.2; input.bands=features(input.timeSeconds);
    auto sought=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    if (!sought || readPixels(sought->frame)!=quietPixels) {
      error="Audio Deformer seek failed to restore the earlier frame"; return false;
    }
  }
  error.clear(); return true;
}
