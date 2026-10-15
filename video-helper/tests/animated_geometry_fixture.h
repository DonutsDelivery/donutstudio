#pragma once

#include "../src/imported_geometry_execution.h"
#include "../src/particle_body_replay.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/VisualStarterModelAssets.h"
#include "multiobject_animation_fixture.h"

inline bool verifyAnimatedObjectGeometryExtraction(std::string& error) {
  using namespace videowire::geometry;
  for (const bool repeated : {false,true}) {
    videohelper::modelpayload::Store store;
    visualanimationimport::ExactContentAssetKey key;
    if (!multiobjectanimationfixture::publish(store,key,false,repeated)) return false;
    visualanimationimport::Request request;
    request.sourceStableId=21; request.deformationStableId=22; request.schedule={21,22};
    request.asset=key; request.sceneIndex=0; request.animationClipStableId=1;
    request.clipName=std::string(visualstartermodel::kClipName);
    const auto source=videohelper::geometry::prepareAnimatedGeometry(store.resolvePreview(key),request,23,error,1);
    if (!source) return false;
    const auto base=*source->descriptor().operations.front().retainedMesh;
    if (base.geometry.positions.size()!=(repeated ? 6u : 9u)
        || base.geometry.vertexIds[0]==base.geometry.vertexIds[3]
        || base.geometry.positions[0].x>=base.geometry.positions[3].x
        || base.attributes.size()!=6 || base.attributes[4].elements[0].components[0]
            ==base.attributes[4].elements[1].components[0]) {
      error="joined imported Geometry must retain separate world-space node and draw identities"; return false;
    }
    PortContract contract; contract.carrier=CarrierKind::geometry3D;
    contract.maxVertices=4096; contract.maxIndices=12288; contract.maxAttributes=kMaximumAttributes;
    const auto reopened=decodeRuntimeValue(encodeRuntimeValue(*source),withAttributeContract(contract,source->descriptor()),{}, {},error);
    if (!reopened) return false;
    videohelper::geometry::ImportedGeometryExecution preview(&store),exported(&store);
    auto first=base,moved=base,reset=base,output=*reopened->descriptor().operations.front().retainedMesh;
    if (!preview.evaluate(first,{0,24000,1001},9,false,error)
        || !preview.evaluate(moved,{12,24000,1001},9,false,error)
        || !preview.evaluate(reset,{0,24000,1001},9,false,error)
        || !exported.evaluate(output,{12,24000,1001},9,true,error)) return false;
    if (equal(first.geometry,moved.geometry) || !equal(first.geometry,reset.geometry)
        || !equal(moved.geometry,output.geometry) || !equalAttributes(moved.attributes,output.attributes)) {
      error="joined native Geometry must animate, rewind and reopen/export with identical topology and attributes"; return false;
    }
    request.pose.nodeStableId=5;
    request.pose.meshStableId=repeated ? 1 : 2;
    const auto selected=videohelper::geometry::prepareAnimatedGeometry(store.resolvePreview(key),request,24,error,2,5);
    if (!selected) return false;
    auto selectedGeometry=*selected->descriptor().operations.front().retainedMesh;
    if (!preview.evaluate(selectedGeometry,{12,24000,1001},9,false,error)) return false;
    if (selectedGeometry.geometry.positions.size()!=(repeated ? 3u : 6u)
        || selectedGeometry.geometry.vertexIds.front()!=base.geometry.vertexIds[3]
        || std::abs(selectedGeometry.geometry.positions.front().x-moved.geometry.positions[3].x)>0.00001) {
      error="Pose Object Geometry must extract the exact repeated node at the same animation time"; return false;
    }
    auto changed=base; changed.geometry.indices.pop_back();
    if (preview.evaluate(changed,{0,24,1},9,false,error) || error.find("topology")==std::string::npos) {
      error="joined imported Geometry must reject topology changes"; return false;
    }
    error.clear();
  }
  return true;
}

inline bool verifyAnimatedGeometryExtraction(std::string& error) {
  using namespace videowire::geometry;
  const std::vector<std::uint8_t> bytes(visualstartermodel::kAnimatedTriangleGlb.begin(),
      visualstartermodel::kAnimatedTriangleGlb.end());
  const visualanimationimport::ExactContentAssetKey key{"animated-geometry-test",1,
      std::string(visualstartermodel::kContentSha256),"model/gltf-binary",bytes.size()};
  std::string encoded;
  constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (std::size_t offset=0;offset<bytes.size();offset+=3) {
    const auto remaining=bytes.size()-offset;
    const auto value=(static_cast<std::uint32_t>(bytes[offset])<<16u)
        | (remaining>1 ? static_cast<std::uint32_t>(bytes[offset+1])<<8u : 0u)
        | (remaining>2 ? bytes[offset+2] : 0u);
    encoded+=alphabet[(value>>18u)&63u]; encoded+=alphabet[(value>>12u)&63u];
    encoded+=remaining>1 ? alphabet[(value>>6u)&63u] : '=';
    encoded+=remaining>2 ? alphabet[value&63u] : '=';
  }
  videohelper::modelpayload::Store store;
  if (!store.begin("animated-geometry-test",key)
      || !store.appendBase64("animated-geometry-test",0,encoded)
      || !store.commit("animated-geometry-test")) {
    error="Animated geometry fixture payload publication failed"; return false;
  }
  visualanimationimport::Request request;
  request.sourceStableId=11; request.deformationStableId=12; request.schedule={11,12};
  request.asset=key; request.sceneIndex=0; request.meshStableId=1;
  request.animationClipStableId=1; request.clipName=std::string(visualstartermodel::kClipName);
  const auto source=videohelper::geometry::prepareAnimatedGeometry(store.resolvePreview(key),request,13,error);
  if (!source) return false;
  const auto base=*source->descriptor().operations[0].retainedMesh;
  videohelper::geometry::ImportedGeometryExecution preview(&store),exported(&store);
  auto first=base, moved=base, repeated=base, saved=base;
  if (!preview.evaluate(first,{0,24,1},1,false,error)
      || !preview.evaluate(moved,{12,24,1},1,false,error)
      || !preview.evaluate(repeated,{0,24,1},1,false,error)
      || !exported.evaluate(saved,{12,24,1},1,true,error)) return false;
  if (equal(first.geometry,moved.geometry) || !equal(first.geometry,repeated.geometry)
      || !equal(moved.geometry,saved.geometry) || !equalAttributes(moved.attributes,saved.attributes)
      || first.geometry.indices!=moved.geometry.indices || first.geometry.vertexIds!=moved.geometry.vertexIds) {
    error="Animated geometry must move, seek exactly, and match preview/export with stable topology"; return false;
  }
  PortContract pointContract; pointContract.carrier=CarrierKind::points3D; pointContract.maxPoints=64;
  pointContract.maxAttributes=kMaximumAttributes;
  const auto points=lowerPointsFromVertices(source->descriptor(),20,pointContract,error);
  if (!points) return false;
  auto binding=std::make_shared<videorender::ParticleGeometryBinding>();
  binding->emitters=std::make_shared<const ValueDescriptor>(*points);
  binding->attractors=binding->emitters;
  binding->mesh=std::make_shared<const ValueDescriptor>(source->descriptor());
  auto history=std::make_shared<videorender::ParticleHistorySource>();
  history->score=std::make_shared<const arbitmod::Score>();
  bool exportParticles=false;
  history->importedGeometry=[&](auto& mesh,const auto& frame,std::uint64_t revision,std::string& failure)
  { return (exportParticles ? exported : preview).evaluate(mesh,frame,revision,exportParticles,failure); };
  videorender::ParticleParams particles;
  particles.motionMode=2; particles.historicalReplay=true; particles.history=history;
  particles.geometryBinding=binding; particles.geometryRevision=3; particles.count=3;
  particles.lifetime=1; particles.force=0.1f; particles.attraction=3;
  const auto sampleParticles=[&](double seconds)
  {
    particles.historyProjectSeconds=seconds;
    return videorender::replayParticleBodies(particles,seconds,1,nullptr,&error);
  };
  const auto bodyStart=sampleParticles(0),bodyMoved=sampleParticles(0.25);
  if (!error.empty() || bodyStart==bodyMoved) { error="Animated imported particle bodies did not move"; return false; }
  sampleParticles(2.1);
  if (!error.empty() || sampleParticles(0.25)!=bodyMoved) { error="Animated imported particle seek changed replay"; return false; }
  exportParticles=true;
  if (sampleParticles(0.25)!=bodyMoved || !error.empty()) { error="Animated imported particle preview/export states differ"; return false; }
  history->importedGeometry=[](auto& mesh,const auto&,std::uint64_t,std::string&)
  { mesh.geometry.indices.pop_back(); return true; };
  sampleParticles(0.25);
  if (error.find("topology")==std::string::npos) { error="Particle animation accepted a topology-changing evaluator"; return false; }
  error.clear();
  PortContract contract; contract.carrier=CarrierKind::geometry3D; contract.maxVertices=4096;
  contract.maxIndices=12288; contract.maxAttributes=kMaximumAttributes;
  auto versionTen=encodeRuntimeValue(*source);
  versionTen[4]=10;
  const auto legacy=decodeRuntimeValue(versionTen,withAttributeContract(contract,source->descriptor()),{}, {},error);
  if (!legacy || legacy->descriptor().operations[0].retainedMesh->importedAnimation!=base.importedAnimation) {
    error="Version 10 imported-animation transport must retain its exact request"; return false;
  }
  Transform transform; transform.translation={1,2,3};
  auto authored=lowerTransformGeometry(source->descriptor(),14,transform,
      withAttributeContract(contract,source->descriptor()),error);
  if (!authored) return false;
  authored->operations[0].retainedMesh=std::make_shared<const RetainedMeshData>(moved);
  ValueDescriptor replayed;
  if (!validateOperationPlan(*authored,{},nullptr,error,&replayed)
      || !admitValue(replayed,withAttributeContract(contract,replayed),error)) return false;
  const auto& position=std::get<GeometryData>(replayed.data).positions.front();
  if (std::abs(position.x-moved.geometry.positions.front().x-1.0f)>0.00001f) {
    error="Geometry operations must replay after GPU animation extraction"; return false;
  }
  auto vectorContract=contract;
  vectorContract.carrier=CarrierKind::field; vectorContract.fieldDomain=Domain::vertex;
  vectorContract.fieldValueType=ValueType::vector; vectorContract.fieldInterpolation=Interpolation::constant;
  vectorContract.maxVertices=vectorContract.maxIndices=vectorContract.maxAttributes=0;
  vectorContract.maxFieldElements=4096;
  auto scalarContract=vectorContract; scalarContract.fieldValueType=ValueType::floatValue;
  const auto positions=lowerElementField(replayed,15,OperationCode::positionField,vectorContract,error);
  FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue;
  settings.mode=3;
  const auto coordinate=positions ? lowerConstructedField(16,OperationCode::fieldVectorMeasure,settings,
      &*positions,nullptr,nullptr,scalarContract,error) : std::nullopt;
  settings.mode=2; settings.values={4,0,0,0,0.25,0,0,0};
  const auto loop=coordinate ? lowerConstructedField(17,OperationCode::fieldTimeline,settings,
      &*coordinate,nullptr,nullptr,scalarContract,error) : std::nullopt;
  settings.mode=1; settings.valueType=ValueType::vector; settings.values={0,1,0,0,0,0,0,0};
  const auto offset=loop ? lowerConstructedField(18,OperationCode::fieldComposeVector,settings,
      &*loop,nullptr,nullptr,vectorContract,error) : std::nullopt;
  const auto combined=offset ? lowerVectorDisplacement(replayed,*offset,19,1,
      withAttributeContract(contract,replayed),error) : std::nullopt;
  if (!combined) return false;
  RuntimeFieldEvaluation timeline; timeline.timelineSeconds=1;
  MaterializedInstancePlan materialized;
  if (!validateOperationPlan(*combined,{},&materialized,error,nullptr,&timeline)) return false;
  const auto& final=materialized.geometries.at(19);
  if (std::abs(final.positions.front().x-position.x)>0.00001f
      || std::abs(final.positions.front().y-position.y-0.25f)>0.00001f
      || final.vertexIds!=std::get<GeometryData>(replayed.data).vertexIds
      || final.indices!=std::get<GeometryData>(replayed.data).indices) {
    error="Timeline fields must preserve the evaluated imported pose and topology"; return false;
  }
  request.pose.meshStableId=1; request.pose.morphEnabled=true;
  request.pose.morphTargetIndex=0; request.pose.morphTargetStableId=1; request.pose.morphWeight=0.75;
  auto posed=base; posed.importedAnimation=visualanimationoperation::encode(request);
  if (!preview.evaluate(posed,{0,24,1},1,false,error) || equal(posed.geometry,first.geometry)) {
    error="Named morph weight must affect extracted GPU geometry"; return false;
  }
  request.pose={};
  request.playback.playback=visualanimation::Playback::Loop;
  auto loopStart=base,looped=base;
  loopStart.importedAnimation=looped.importedAnimation=visualanimationoperation::encode(request);
  videohelper::ImportedAnimationDeformationConsumer decoded;
  if (!decoded.admit(request,bytes.data(),bytes.size(),error,true)) return false;
  const auto duration=decoded.admittedDocument()->clips[0].clip->durationSeconds();
  if (!preview.evaluate(loopStart,{0,24,1},2,false,error)
      || !preview.evaluate(looped,{static_cast<std::int64_t>(std::llround(duration*24)),24,1},2,false,error)
      || !equal(loopStart.geometry,looped.geometry)) {
    error="Animated geometry loop and reset must return the authored pose"; return false;
  }
  return verifyAnimatedObjectGeometryExtraction(error);
}
