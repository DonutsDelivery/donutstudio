#pragma once

#include "../src/geometry_core_backend.h"
#include "../src/geometry_score_field.h"

inline std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> scoreFieldFrame(
    bool reverse=false, bool empty=false, bool links=true, float velocity=127.0f) {
  auto score=std::make_shared<arbitmod::Score>();
  score->notationVersion=1; score->scoreRevision=1; score->rootFreq=440;
  score->historyBeats=8; score->lookaheadBeats=8;
  if (!empty) for (int index=0;index<2;++index) {
    arbitmod::Note note;
    note.id=index==0 ? -1 : -102; note.startBeat=static_cast<float>(index*2);
    note.lengthBeats=8; note.velocity=velocity; note.freqHz=index==0 ? 440.0f : 880.0f;
    note.trackId=0; note.durationSeconds=4;
    for (std::size_t prime=0;prime<6;++prime) note.primes[prime]=static_cast<float>(prime+1);
    if (index==1) note.linkMasterId=-1;
    score->notes.push_back(note);
  }
  if (reverse) std::reverse(score->notes.begin(),score->notes.end());
  if (!empty && links) score->links.push_back({-301,-102,-1,2,1,0});
  canonicalblockc::FrameKey key;
  key.projectGeneration=key.sourceGeneration=key.helperGeneration=1;
  key.backendGeneration=key.deviceGeneration=key.scoreGeneration=1;
  key.beatMapGeneration=key.fpsGeneration=key.loopGeneration=key.seekGeneration=1;
  key.fps=60;
  canonicalblockc::FrameProducer producer;
  return producer.evaluate(key,score,0);
}

template <typename ReadPixels>
bool verifyNativeScoreFields(arbitgpu::NativeFixtureSceneBackend& backend,
                             ReadPixels readPixels,std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  PortContract contract;
  contract.carrier=CarrierKind::geometry3D; contract.overflow=OverflowPolicy::reject;
  contract.maxVertices=4; contract.maxIndices=6;
  const auto mesh=lowerGrid(951,951,1,2,2,1,contract,error);
  if (!mesh) return false;
  const auto original=admitValue(*mesh,contract,error);
  if (!original) return false;
  GeometryCorePlanRuntime runtime(capabilities);
  const auto originalPlan=runtime.admitPreview({1,1,1,950,1,PlanUse::preview},
      lowerRuntimePlan(contract,*original),{}, {},error);
  if (!originalPlan) return false;
  const auto originalDraw=executeNativeGeometry(backend,*originalPlan,64,64,error);
  if (!originalDraw) return false;
  const auto before=readPixels(originalDraw->frame);
  arbitgpu::NativeFixtureSceneRuntimeInputs ambientScoreInputs;
  ambientScoreInputs.canonicalBlockCFrame=scoreFieldFrame();
  const auto ambientScoreDraw=executeNativeGeometry(backend,*originalPlan,64,64,error,false,ambientScoreInputs);
  if (!ambientScoreDraw || readPixels(ambientScoreDraw->frame)!=before
      || ambientScoreDraw->stats.noteInstanceDrawCount!=0) {
    error="A canonical score frame alone must not reject or instance a static Geometry Core draw";
    return false;
  }
  for (unsigned mode=0;mode<5;++mode) {
    scorefield::Binding binding;
    binding.fieldStableId=952; binding.sourceGeometryStableId=951;
    binding.consumerGeometryStableId=953; binding.scoreSourceStableId=954;
    binding.mode=static_cast<scorefield::Mode>(mode); binding.radius=4;
    binding.direction={0.25f,0,0}; binding.gain=mode==2 ? 0.1f : 1;
    for (const auto& p : std::get<GeometryData>(mesh->data).positions)
      binding.positions.push_back({p.x,p.y,p.z});
    binding.elementIds=std::get<GeometryData>(mesh->data).vertexIds;
    auto value=*mesh;
    value.stableId=953; value.scoreFields.push_back(binding);
    value.operations.push_back({952,OperationCode::evaluateField,952,952,{}});
    value.operations.push_back({953,OperationCode::transformTRS,951,953,{}});
    value.dispatchCount+=2;
    const auto admitted=admitValue(value,contract,error);
    if (!admitted) return false;
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    GeometryCorePlanRuntime fieldRuntime(capabilities);
    const auto preview=fieldRuntime.admitPreview({1,1,1,953,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=fieldRuntime.admitExport({1,1,1,953,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.canonicalBlockCFrame=scoreFieldFrame();
    const auto previewDraw=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
    inputs.canonicalBlockCFrame=scoreFieldFrame(true);
    const auto exportDraw=executeNativeGeometry(backend,*exported,64,64,error,false,inputs);
    if (!previewDraw || !exportDraw) return false;
    const auto pixels=readPixels(previewDraw->frame);
    if (pixels.empty() || pixels==before || pixels!=readPixels(exportDraw->frame)
        || previewDraw->stats.ordinaryDrawCount!=1 || previewDraw->stats.noteInstanceDrawCount!=0
        || exportDraw->stats.ordinaryDrawCount!=1 || exportDraw->stats.noteInstanceDrawCount!=0) {
      error="Score Field must change native pixels and preserve preview/export equality under score reordering";
      return false;
    }
    if (mode==1) {
      inputs.canonicalBlockCFrame=scoreFieldFrame(false,false,true,32);
      const auto softer=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
      if (!softer || readPixels(softer->frame)==pixels) {
        error="Velocity Influence did not respond to the canonical note velocity"; return false;
      }
      inputs.canonicalBlockCFrame=scoreFieldFrame();
      const auto repeated=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
      if (!repeated || readPixels(repeated->frame)!=pixels) {
        error="Score Field retained history after returning to an earlier score frame"; return false;
      }
    }
    if (executeNativeGeometry(backend,*preview,64,64,error).has_value()
        || error.find("canonical score frame")==std::string::npos) {
      error="Score Field rendered without a canonical score frame"; return false;
    }
  }
  error.clear(); return true;
}
