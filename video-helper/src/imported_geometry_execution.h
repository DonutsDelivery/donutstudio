#pragma once

#include "glb_geometry_core_adapter.h"
#include "glb_scene_adapter.h"
#include "imported_animation_deformation_consumer.h"
#include "model_payload_transport.h"
#include "gpu_backend/backend.h"
#include "../../shared/VisualImportedAnimationOperationContract.h"

namespace videohelper::geometry {

// Static admission retains base topology. Every animated evaluation resolves
// the same exact asset and obtains positions from native GPU deformation.
inline std::optional<videowire::geometry::AdmittedValue> prepareAnimatedGeometry(
    const visualmodelassetpayload::PayloadPtr& payload,
    const visualanimationimport::Request& request, std::uint64_t geometryId,
    std::string& error, std::uint32_t objectScope=0, std::uint32_t objectNodeStableId=0) {
  if (!payload || !visualanimationimport::sameAsset(payload->key(),request.asset)
      || (request.meshStableId==0 && objectScope==0) || objectScope>2
      || (objectScope==2 && (objectNodeStableId==0 || objectNodeStableId!=request.pose.nodeStableId))) {
    error="Animated Geometry3D requires the exact selected asset and mesh"; return std::nullopt;
  }
  ImportedAnimationDeformationConsumer consumer;
  if (!consumer.admit(request,payload->bytes().data(),payload->bytes().size(),error,objectScope==0)) return std::nullopt;
  auto options=nativeImportedAnimationDecodeOptions(payload->bytes().size(),request.sceneIndex);
  const auto base=gltf::decodeAnimatedGlbBaseScene(payload->bytes(),options.admission,error);
  const auto meshIndex=static_cast<std::size_t>(request.meshStableId-1u);
  if (!base || (request.meshStableId!=0 && meshIndex>=base->meshes.size())) return std::nullopt;
  if (objectScope==0 && base->meshes[meshIndex].primitives.size()!=1) {
    error="Animated Geometry3D requires one triangle primitive with stable topology"; return std::nullopt;
  }
  const auto selected=request.meshStableId==0 ? std::nullopt : std::optional<std::size_t>{meshIndex};
  const auto native=gltf::adaptAnimatedGlbMeshToVisual3DScene(*base,error,selected,objectScope!=0);
  if (!native) return std::nullopt;
  auto admitted=objectScope==0
      ? gltf::adaptGlbMeshToGeometryCore(*base,meshIndex,geometryId,request.sourceStableId,request.asset.version,error,true)
      : gltf::adaptGlbObjectsToGeometryCore(*base,selected,objectScope==2 ? objectNodeStableId : 0,
          geometryId,request.sourceStableId,request.asset.version,error);
  if (!admitted) return std::nullopt;
  auto value=admitted->descriptor();
  auto retained=std::make_shared<videowire::geometry::RetainedMeshData>(*value.operations.front().retainedMesh);
  retained->importedAnimation=visualanimationoperation::encode(request);
  value.operations.front().retainedMesh=std::move(retained);
  videowire::geometry::PortContract contract;
  contract.carrier=videowire::geometry::CarrierKind::geometry3D;
  contract.maxVertices=4096; contract.maxIndices=12288;
  contract.maxAttributes=videowire::geometry::kMaximumAttributes;
  return videowire::geometry::admitValue(std::move(value),
      videowire::geometry::withAttributeContract(contract,admitted->descriptor()),{},error);
}

class ImportedGeometryExecution final {
public:
  explicit ImportedGeometryExecution(modelpayload::Store* store) : store_(store) {}

  void releaseNativeResources() noexcept {
    resources_.reset();
    identity_.clear();
  }

  bool evaluate(videowire::geometry::RetainedMeshData& geometry,
      const visualdeformation::RationalFrameTime& frame, std::uint64_t revision,
      bool exporting, std::string& error) {
    visualanimationimport::Request request;
    if (!store_ || frame.rateNumerator==0 || frame.rateDenominator==0 || revision==0) {
      error="Animated Geometry3D has no exact asset, animation request or frame identity"; return false;
    }
    if (!visualanimationoperation::decode(geometry.importedAnimation,request,error)) return false;
    const auto payload=exporting ? store_->resolveExport(request.asset) : store_->resolvePreview(request.asset);
    if (!payload) { error="Animated Geometry3D exact asset bytes are unavailable"; return false; }
    if (std::any_of(geometry.attributes.begin(),geometry.attributes.end(),
        [](const auto& attribute) { return attribute.descriptor.name=="importedObjectId"; }))
      return evaluateObjects(geometry,request,payload,frame,revision,exporting,error);
    const auto identity=geometry.importedAnimation+"\n"+std::to_string(revision);
    if (identity!=identity_ || payload_!=payload) {
      ImportedAnimationDeformationConsumer consumer;
      if (!consumer.admit(request,payload->bytes().data(),payload->bytes().size(),error,true)) return false;
      const auto document=consumer.admittedDocument();
      const auto binding=std::find_if(document->renderBindings.begin(),document->renderBindings.end(),
          [&](const auto& b) { return b.mesh.value==request.meshStableId; });
      auto options=nativeImportedAnimationDecodeOptions(payload->bytes().size(),request.sceneIndex);
      const auto base=gltf::decodeAnimatedGlbBaseScene(payload->bytes(),options.admission,error);
      const auto selected=static_cast<std::size_t>(request.meshStableId-1u);
      if (!base || selected>=base->meshes.size()
          || base->meshes[selected].primitives.size()!=1) {
        error="Animated Geometry3D requires one selected mesh and one stable triangle primitive"; return false;
      }
      auto scene=gltf::adaptAnimatedGlbMeshToVisual3DScene(*base,error,selected);
      if (!scene) return false;
      ImportedAnimationDeformationEvaluation initial;
      if (!consumer.evaluatePreview(request,frame,revision,initial,error)) return false;
      auto source=std::make_shared<arbitgpu::NativeDeformationScene>();
      source->sourceStableId=request.sourceStableId;
      source->deformationStableId=request.deformationStableId;
      source->structuralRevision=revision;
      source->clip=initial.deformation ? initial.deformation->clip() : initial.sceneAnimation->clip();
      source->object=scene->objects[0].id;
      source->scene=std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(std::move(*scene));
      source->retainDeformedGeometry=true;
      std::shared_ptr<const arbitgpu::NativeDeformationResources> resources;
      if (binding!=document->renderBindings.end()) {
        source->mesh=binding->mesh; source->skin=binding->skin;
        source->animationNodeStableId=binding->nodeIndex+1u;
        source->deformation=document->deformation; source->morphBaseWeights=binding->morphBaseWeights;
        for (const auto& joint : binding->jointBaseTransforms)
          source->jointBaseTransforms.push_back({joint.skin,joint.joint,joint.translation,joint.rotation,joint.scale,joint.matrix});
        auto prepared=arbitgpu::nativeDeformationBackend().prepare(source);
        if (!prepared.prepared) { error=prepared.error; return false; }
        resources=std::move(prepared.resources);
      }
      identity_=identity; payload_=payload; consumer_=std::move(consumer);
      source_=std::move(source); resources_=std::move(resources);
    }
    request.playback.timelineSeconds += static_cast<double>(frame.frame)*frame.rateDenominator/frame.rateNumerator;
    const auto& scene=*source_->scene;
    if (geometry.geometry.positions.size()!=scene.vertexCount
        || geometry.geometry.indices.size()!=scene.indexCount
        || !std::equal(geometry.geometry.indices.begin(),geometry.geometry.indices.end(),scene.indices.begin())) {
      error="Animated Geometry3D topology does not match the exact selected asset"; return false;
    }
    for (std::size_t index=0;index<geometry.geometry.vertexIds.size();++index)
      if (geometry.geometry.vertexIds[index]!=request.meshStableId*65536u+index+1u) {
        error="Animated Geometry3D vertex identities do not match the exact selected mesh"; return false;
      }
    ImportedAnimationDeformationEvaluation evaluation;
    if (!(exporting ? consumer_.evaluateExport(request,frame,revision,evaluation,error)
                    : consumer_.evaluatePreview(request,frame,revision,evaluation,error))) return false;
    arbitgpu::NativeDeformationRuntimeInputs inputs;
    inputs.morphWeight=request.pose.morphEnabled || request.combinationMode==visualanimation::CombinationMode::WeightedBlend
        ? 1.0f : static_cast<float>(request.playback.weight);
    arbitgpu::NativeDeformationSubmission rendered;
    if (resources_) {
      rendered=arbitgpu::nativeDeformationBackend().render(source_,evaluation.deformation,resources_,1,1,inputs);
      if (!rendered.rendered) { error=rendered.error; return false; }
    } else for (std::size_t index=0;index<scene.vertexCount;++index) {
      const auto& v=scene.vertices[index];
      rendered.deformedVertices.insert(rendered.deformedVertices.end(),
          {v.position.x,v.position.y,v.position.z,v.normal.x,v.normal.y,v.normal.z,v.uv.x,v.uv.y});
    }
    if (rendered.deformedVertices.size()!=geometry.geometry.positions.size()*8u) {
      error="Animated Geometry3D changed the selected mesh topology"; return false;
    }
    auto normals=std::find_if(geometry.attributes.begin(),geometry.attributes.end(),
        [](const auto& attribute) { return attribute.descriptor.name=="normal"; });
    if (normals==geometry.attributes.end() || normals->elements.size()!=geometry.geometry.positions.size()) {
      error="Animated Geometry3D lost its vertex normal attribute"; return false;
    }
    for (std::size_t index=0;index<geometry.geometry.positions.size();++index) {
      const auto* values=rendered.deformedVertices.data()+index*8u;
      if (!std::all_of(values,values+8,[](float v) { return std::isfinite(v); })) {
        error="Animated Geometry3D produced non-finite GPU vertices"; return false;
      }
      geometry.geometry.positions[index]={values[0],values[1],values[2]};
      normals->elements[index].components={values[3],values[4],values[5],0};
    }
    return true;
  }

private:
  bool evaluateObjects(videowire::geometry::RetainedMeshData& geometry,
      visualanimationimport::Request request, const visualmodelassetpayload::PayloadPtr& payload,
      const visualdeformation::RationalFrameTime& frame, std::uint64_t revision,
      bool exporting, std::string& error) {
    const auto nodes=std::find_if(geometry.attributes.begin(),geometry.attributes.end(),
        [](const auto& attribute) { return attribute.descriptor.name=="importedNodeId"; });
    if (nodes==geometry.attributes.end() || nodes->elements.empty()
        || nodes->elements.size()!=geometry.geometry.positions.size()) {
      error="Imported object geometry lost its exact node identities"; return false;
    }
    const auto firstNode=nodes->elements.front().components[0];
    if (!std::isfinite(firstNode) || firstNode<1 || firstNode>65536 || std::floor(firstNode)!=firstNode) {
      error="Imported object geometry node identity is invalid"; return false;
    }
    const auto nodeId=std::all_of(nodes->elements.begin(),nodes->elements.end(),
        [&](const auto& node) { return node.components[0]==firstNode; }) ? static_cast<std::uint32_t>(firstNode) : 0u;
    const auto identity="objects\n"+geometry.importedAnimation+"\n"+std::to_string(revision)+"\n"+std::to_string(nodeId);
    if (identity!=identity_ || payload_!=payload) {
      ImportedAnimationDeformationConsumer consumer;
      if (!consumer.admit(request,payload->bytes().data(),payload->bytes().size(),error)) return false;
      const auto document=consumer.admittedDocument();
      auto options=nativeImportedAnimationDecodeOptions(payload->bytes().size(),request.sceneIndex);
      auto base=gltf::decodeAnimatedGlbBaseScene(payload->bytes(),options.admission,error);
      if (!base) return false;
      const auto selected=request.meshStableId==0 ? std::nullopt
          : std::optional<std::size_t>{static_cast<std::size_t>(request.meshStableId-1u)};
      const auto expected=gltf::adaptGlbObjectsToGeometryCore(*base,selected,nodeId,1,request.sourceStableId,
          request.asset.version,error);
      auto scene=gltf::adaptAnimatedGlbMeshToVisual3DScene(*base,error,selected,true);
      if (!expected || !scene) return false;
      ImportedAnimationDeformationEvaluation initial;
      if (!consumer.evaluatePreview(request,frame,revision,initial,error)) return false;
      auto source=std::make_shared<arbitgpu::NativeDeformationScene>();
      source->sourceStableId=request.sourceStableId; source->deformationStableId=request.deformationStableId;
      source->structuralRevision=revision;
      source->clip=initial.deformation ? initial.deformation->clip() : initial.sceneAnimation->clip();
      source->scene=std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(std::move(*scene));
      source->deformation=document->deformation; source->retainDeformedGeometry=true;
      for (const auto& binding : document->renderBindings) {
        const auto object=std::find_if(source->scene->objects.begin(),source->scene->objects.begin()+source->scene->objectCount,
            [&](const auto& value) { return value.id.value==binding.nodeIndex*65536u+1u; });
        if (object==source->scene->objects.begin()+source->scene->objectCount
            || (nodeId!=0 && binding.nodeIndex+1u!=nodeId)) continue;
        auto draw=std::make_shared<arbitgpu::NativeDeformationScene>(*source);
        draw->draws.clear(); draw->mesh=binding.mesh; draw->object=object->id; draw->skin=binding.skin;
        draw->animationNodeStableId=binding.nodeIndex+1u; draw->morphBaseWeights=binding.morphBaseWeights;
        draw->batchMember=true;
        for (const auto& joint : binding.jointBaseTransforms)
          draw->jointBaseTransforms.push_back({joint.skin,joint.joint,joint.translation,joint.rotation,joint.scale,joint.matrix});
        source->draws.push_back(std::move(draw));
      }
      std::shared_ptr<const arbitgpu::NativeDeformationResources> resources;
      if (!source->draws.empty()) {
        source->mesh=source->draws.front()->mesh; source->object=source->draws.front()->object;
        auto prepared=arbitgpu::nativeDeformationBackend().prepare(source);
        if (!prepared.prepared) { error=prepared.error; return false; }
        resources=std::move(prepared.resources);
      }
      objectVertices_.clear();
      for (std::size_t draw=0;draw<source->scene->objectCount;++draw) {
        const auto& object=source->scene->objects[draw]; const auto node=(object.id.value-1u)/65536u;
        if (nodeId!=0 && node+1u!=nodeId) continue;
        for (std::size_t vertex=0;vertex<object.vertexCount;++vertex)
          objectVertices_.push_back({object.firstVertex+vertex,node});
      }
      expectedObjects_=expected->descriptor().operations.front().retainedMesh;
      objectBase_=std::make_shared<const gltf::GlbStaticMeshDocument>(std::move(*base));
      identity_=identity; payload_=payload; consumer_=std::move(consumer);
      source_=std::move(source); resources_=std::move(resources);
    }
    if (geometry.geometry.indices!=expectedObjects_->geometry.indices
        || geometry.geometry.vertexIds!=expectedObjects_->geometry.vertexIds
        || geometry.geometry.positions.size()!=objectVertices_.size()) {
      error="Imported object geometry topology or stable vertex identities changed"; return false;
    }
    for (const auto* name : {"importedNodeId","importedObjectId","materialIndex"}) {
      const auto actual=std::find_if(geometry.attributes.begin(),geometry.attributes.end(),
          [&](const auto& attribute) { return attribute.descriptor.name==name; });
      const auto expected=std::find_if(expectedObjects_->attributes.begin(),expectedObjects_->attributes.end(),
          [&](const auto& attribute) { return attribute.descriptor.name==name; });
      if (actual==geometry.attributes.end() || actual->elements.size()!=expected->elements.size()
          || !std::equal(actual->elements.begin(),actual->elements.end(),expected->elements.begin(),
              [](const auto& a,const auto& b) { return a.components==b.components; })) {
        error="Imported object geometry changed its node, draw or material identities"; return false;
      }
    }
    request.playback.timelineSeconds+=static_cast<double>(frame.frame)*frame.rateDenominator/frame.rateNumerator;
    ImportedAnimationDeformationEvaluation evaluation;
    if (!(exporting ? consumer_.evaluateExport(request,frame,revision,evaluation,error)
                    : consumer_.evaluatePreview(request,frame,revision,evaluation,error))) return false;
    std::vector<float> vertices;
    if (!source_->draws.empty()) {
      arbitgpu::NativeDeformationRuntimeInputs inputs;
      inputs.morphWeight=request.pose.morphEnabled || request.combinationMode==visualanimation::CombinationMode::WeightedBlend
          ? 1.0f : static_cast<float>(request.playback.weight);
      auto rendered=arbitgpu::nativeDeformationBackend().render(source_,evaluation.deformation,resources_,1,1,inputs);
      if (!rendered.rendered) { error=rendered.error; return false; }
      if (rendered.deformedVertices.size()!=source_->scene->vertexCount*8u) {
        error="Imported object geometry GPU readback changed topology"; return false;
      }
      vertices=std::move(rendered.deformedVertices);
    }
    auto normals=std::find_if(geometry.attributes.begin(),geometry.attributes.end(),
        [](const auto& attribute) { return attribute.descriptor.name=="normal"; });
    if (normals==geometry.attributes.end() || normals->elements.size()!=objectVertices_.size()) {
      error="Imported object geometry lost its normal attribute"; return false;
    }
    for (std::size_t index=0;index<objectVertices_.size();++index) {
      const auto [vertex,node]=objectVertices_[index];
      const auto& base=source_->scene->vertices[vertex];
      auto& position=geometry.geometry.positions[index]; auto& normal=normals->elements[index];
      if (vertices.empty()) {
        position={base.position.x,base.position.y,base.position.z};
        normal.components={base.normal.x,base.normal.y,base.normal.z,0};
      } else {
        const auto* values=vertices.data()+vertex*8u;
        position={values[0],values[1],values[2]}; normal.components={values[3],values[4],values[5],0};
      }
      if (!gltf::transformGlbGeometryVertex(position,normal,objectBase_->nodes[node].worldTransform)) {
        error="Imported object geometry produced non-finite world-space GPU vertices"; return false;
      }
    }
    return true;
  }

  modelpayload::Store* store_=nullptr;
  std::string identity_;
  visualmodelassetpayload::PayloadPtr payload_;
  ImportedAnimationDeformationConsumer consumer_;
  std::shared_ptr<const arbitgpu::NativeDeformationScene> source_;
  std::shared_ptr<const arbitgpu::NativeDeformationResources> resources_;
  std::shared_ptr<const gltf::GlbStaticMeshDocument> objectBase_;
  std::shared_ptr<const videowire::geometry::RetainedMeshData> expectedObjects_;
  std::vector<std::pair<std::size_t,std::size_t>> objectVertices_;
};
} // namespace videohelper::geometry
