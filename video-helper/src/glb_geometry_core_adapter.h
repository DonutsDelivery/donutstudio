#pragma once

#include "gltf_glb.h"
#include "glb_scene_adapter.h"
#include "../../shared/GeometryCoreTransport.h"
#include <map>

namespace videohelper::gltf {

inline bool transformGlbGeometryVertex(videowire::geometry::Vec3& point,
    videowire::geometry::FieldElement& normal, const std::array<float,16>& m) {
  const auto p=point;
  point={m[0]*p.x+m[4]*p.y+m[8]*p.z+m[12],m[1]*p.x+m[5]*p.y+m[9]*p.z+m[13],
      m[2]*p.x+m[6]*p.y+m[10]*p.z+m[14]};
  const auto& n=normal.components;
  const auto determinant=m[0]*(m[5]*m[10]-m[9]*m[6])-m[4]*(m[1]*m[10]-m[9]*m[2])
      +m[8]*(m[1]*m[6]-m[5]*m[2]);
  if (!std::isfinite(determinant) || std::abs(determinant)<1e-20) return false;
  const double x=((m[5]*m[10]-m[9]*m[6])*n[0]+(m[9]*m[2]-m[1]*m[10])*n[1]+(m[1]*m[6]-m[5]*m[2])*n[2])/determinant;
  const double y=((m[8]*m[6]-m[4]*m[10])*n[0]+(m[0]*m[10]-m[8]*m[2])*n[1]+(m[4]*m[2]-m[0]*m[6])*n[2])/determinant;
  const double z=((m[4]*m[9]-m[8]*m[5])*n[0]+(m[8]*m[1]-m[0]*m[9])*n[1]+(m[0]*m[5]-m[4]*m[1])*n[2])/determinant;
  const auto length=std::sqrt(x*x+y*y+z*z);
  if (!videowire::geometry::finite(point) || !std::isfinite(length) || length==0) return false;
  normal.components={x/length,y/length,z/length,0}; return true;
}

// Scene Objects and Pose Object use joined world-space geometry. Named face
// attributes retain exact imported draw/material IDs, including repeated meshes.
inline std::optional<videowire::geometry::AdmittedValue> adaptGlbObjectsToGeometryCore(
    const GlbStaticMeshDocument& asset, std::optional<std::size_t> meshIndex,
    std::uint32_t nodeStableId, videowire::geometry::StableId valueId,
    videowire::geometry::StableId sourceId, std::uint64_t revision, std::string& error) {
  using namespace videowire::geometry;
  const auto scene=adaptAnimatedGlbMeshToVisual3DScene(asset,error,meshIndex,true);
  if (!scene) return std::nullopt;
  auto retained=std::make_shared<RetainedMeshData>();
  retained->attributes={
      {{1,"normal",ValueType::vector,Domain::vertex,Interpolation::normalizedLinear},{}},
      {{2,"uv",ValueType::vector,Domain::vertex,Interpolation::linear},{}},
      {{3,"color",ValueType::color,Domain::vertex,Interpolation::linear},{}},
      {{4,"materialIndex",ValueType::integer,Domain::face,Interpolation::nearest},{}},
      {{5,"importedObjectId",ValueType::integer,Domain::face,Interpolation::nearest},{}},
      {{6,"importedNodeId",ValueType::integer,Domain::vertex,Interpolation::nearest},{}}};
  auto& mesh=retained->geometry;
  std::map<std::size_t,std::size_t> nodeVertices;
  for (std::size_t draw=0;draw<scene->objectCount;++draw) {
    const auto& object=scene->objects[draw];
    const auto node=(object.id.value-1u)/65536u;
    if (nodeStableId!=0 && node+1u!=nodeStableId) continue;
    const auto base=mesh.positions.size(), localBase=nodeVertices[node];
    for (std::size_t vertex=0;vertex<object.vertexCount;++vertex) {
      const auto& source=scene->vertices[object.firstVertex+vertex];
      Vec3 p{source.position.x,source.position.y,source.position.z};
      FieldElement normal{{source.normal.x,source.normal.y,source.normal.z,0}};
      if (!transformGlbGeometryVertex(p,normal,asset.nodes[node].worldTransform)) {
        error="Imported object geometry has a singular or non-finite world transform"; return std::nullopt;
      }
      mesh.positions.push_back(p); mesh.vertexIds.push_back((node+1u)*65536ull+localBase+vertex+1u);
      retained->attributes[0].elements.push_back(normal);
      retained->attributes[1].elements.push_back({{source.uv.x,source.uv.y,0,0}});
      retained->attributes[2].elements.push_back({{source.color[0],source.color[1],source.color[2],source.color[3]}});
      retained->attributes[5].elements.push_back({{static_cast<double>(node+1u),0,0,0}});
    }
    nodeVertices[node]+=object.vertexCount;
    for (std::size_t index=0;index<object.indexCount;++index)
      mesh.indices.push_back(static_cast<std::uint32_t>(base+scene->indices[object.firstIndex+index]));
    for (std::size_t face=0;face<object.indexCount/3;++face) {
      retained->attributes[3].elements.push_back({{static_cast<double>(object.material.value-1u),0,0,0}});
      retained->attributes[4].elements.push_back({{static_cast<double>(object.id.value),0,0,0}});
    }
  }
  if (mesh.positions.empty()) { error="Selected imported object is not drawn by this scene and mesh"; return std::nullopt; }
  ValueDescriptor value; value.carrier=CarrierKind::geometry3D; value.stableId=valueId;
  value.sourceStableId=sourceId; value.sourceRevision=revision; value.dispatchCount=1;
  value.data=mesh; value.attributes=retained->attributes;
  OperationRecord operation{valueId,OperationCode::retainedMesh,sourceId,valueId,{}};
  operation.retainedMesh=std::move(retained); value.operations.push_back(std::move(operation));
  PortContract contract; contract.carrier=CarrierKind::geometry3D;
  contract.maxVertices=4096; contract.maxIndices=12288; contract.maxAttributes=kMaximumAttributes;
  contract=withAttributeContract(contract,value);
  return admitValue(std::move(value),contract,{},error);
}

// Extract mesh-local geometry from the existing admitted decoder. Asset and
// scene selection stay with the Imported Model source in the authored graph.
inline std::optional<videowire::geometry::AdmittedValue> adaptGlbMeshToGeometryCore(
    const GlbStaticMeshDocument& asset, std::size_t meshIndex,
    videowire::geometry::StableId valueId, videowire::geometry::StableId sourceId,
    std::uint64_t revision, std::string& error, bool animatedBase = false) {
  using namespace videowire::geometry;
  if (!animatedBase && (asset.metadata.animations!=0 || asset.metadata.skins!=0)) {
    error="Imported Mesh Geometry requires a static mesh; animated and skinned extraction is unavailable";
    return std::nullopt;
  }
  if (meshIndex>=asset.meshes.size() || asset.selectedScene>=asset.scenes.size()) {
    error="Imported Mesh Geometry selection is out of range"; return std::nullopt;
  }
  std::set<std::size_t> visited;
  auto pending=asset.scenes[asset.selectedScene].rootNodes;
  bool selected=false;
  while (!pending.empty()) {
    const auto index=pending.back(); pending.pop_back();
    if (index>=asset.nodes.size() || !visited.insert(index).second) {
      error="Imported mesh scene has invalid or repeated nodes"; return std::nullopt;
    }
    const auto& node=asset.nodes[index];
    selected=selected || (node.mesh && *node.mesh==meshIndex);
    pending.insert(pending.end(),node.children.begin(),node.children.end());
  }
  if (!selected) { error="Selected mesh is not present in the selected scene"; return std::nullopt; }
  auto retained=std::make_shared<RetainedMeshData>();
  retained->attributes={
      {{1,"normal",ValueType::vector,Domain::vertex,Interpolation::normalizedLinear},{}},
      {{2,"uv",ValueType::vector,Domain::vertex,Interpolation::linear},{}},
      {{3,"color",ValueType::color,Domain::vertex,Interpolation::linear},{}},
      {{4,"materialIndex",ValueType::integer,Domain::face,Interpolation::nearest},{}}};
  auto& mesh=retained->geometry;
  for (const auto& primitive : asset.meshes[meshIndex].primitives) {
    const auto count=primitive.positions.size()/3, base=mesh.positions.size();
    if (primitive.positions.size()%3!=0 || count==0 || count>4096-base
        || primitive.indices.empty() || primitive.indices.size()%3!=0
        || primitive.indices.size()>12288-mesh.indices.size()
        || (!primitive.normals.empty() && primitive.normals.size()!=count*3)
        || (!primitive.texCoords0.empty() && primitive.texCoords0.size()!=count*2)
        || (!primitive.colors0.empty() && primitive.colors0.size()!=count*4)
        || (primitive.material && *primitive.material>=asset.materials.size())) {
      error="Imported mesh exceeds 4096 vertices or 12288 indices, or has incomplete attributes";
      return std::nullopt;
    }
    for (std::size_t i=0;i<count;++i) {
      mesh.positions.push_back({primitive.positions[i*3],primitive.positions[i*3+1],primitive.positions[i*3+2]});
      // Mesh index and the concatenated primitive vertex offset match the
      // decoder's mesh ordering. IDs do not change with graph transforms.
      mesh.vertexIds.push_back((static_cast<StableId>(meshIndex)+1u)*65536u+base+i+1u);
      FieldElement normal{{0,0,0,0}}, uv{{0,0,0,0}}, color{{1,1,1,1}};
      if (!primitive.normals.empty()) for (std::size_t axis=0;axis<3;++axis) normal.components[axis]=primitive.normals[i*3+axis];
      if (!primitive.texCoords0.empty()) for (std::size_t axis=0;axis<2;++axis) uv.components[axis]=primitive.texCoords0[i*2+axis];
      if (!primitive.colors0.empty()) for (std::size_t axis=0;axis<4;++axis) color.components[axis]=primitive.colors0[i*4+axis];
      retained->attributes[0].elements.push_back(normal);
      retained->attributes[1].elements.push_back(uv);
      retained->attributes[2].elements.push_back(color);
    }
    for (const auto index : primitive.indices) {
      if (index>=count) { error="Imported mesh index is out of range"; return std::nullopt; }
      mesh.indices.push_back(static_cast<std::uint32_t>(base+index));
    }
    for (std::size_t triangle=0;triangle<primitive.indices.size();triangle+=3) {
      retained->attributes[3].elements.push_back({{primitive.material ? static_cast<double>(*primitive.material) : -1.0,0,0,0}});
      if (primitive.normals.empty()) {
        const auto a=base+primitive.indices[triangle],b=base+primitive.indices[triangle+1],c=base+primitive.indices[triangle+2];
        const auto& p=mesh.positions[a]; const auto& q=mesh.positions[b]; const auto& r=mesh.positions[c];
        const std::array<double,3> n{{(q.y-p.y)*(r.z-p.z)-(q.z-p.z)*(r.y-p.y),
            (q.z-p.z)*(r.x-p.x)-(q.x-p.x)*(r.z-p.z),(q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x)}};
        for (auto vertex : {a,b,c}) for (std::size_t axis=0;axis<3;++axis)
          retained->attributes[0].elements[vertex].components[axis]+=n[axis];
      }
    }
  }
  for (auto& element : retained->attributes[0].elements) {
    auto& n=element.components;
    const auto length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    if (!std::isfinite(length)) { error="Imported mesh normal is non-finite"; return std::nullopt; }
    if (length>0) for (std::size_t axis=0;axis<3;++axis) n[axis]/=length;
    else n={0,0,1,0};
  }
  ValueDescriptor value;
  value.carrier=CarrierKind::geometry3D; value.stableId=valueId;
  value.sourceStableId=sourceId; value.sourceRevision=revision; value.dispatchCount=1;
  value.data=mesh; value.attributes=retained->attributes;
  OperationRecord operation{valueId,OperationCode::retainedMesh,sourceId,valueId,{}};
  operation.retainedMesh=std::move(retained); value.operations.push_back(std::move(operation));
  PortContract contract; contract.carrier=CarrierKind::geometry3D;
  contract.maxVertices=4096; contract.maxIndices=12288; contract.maxAttributes=kMaximumAttributes;
  contract=withAttributeContract(std::move(contract),value);
  return admitValue(std::move(value),contract,{},error);
}

inline std::optional<videowire::geometry::AdmittedValue> decodeGlbMeshToGeometryCore(
    const std::vector<std::uint8_t>& bytes, std::optional<std::size_t> sceneIndex,
    std::size_t meshIndex, videowire::geometry::StableId valueId,
    videowire::geometry::StableId sourceId, std::uint64_t revision, std::string& error) {
  GlbAdmissionOptions options; options.sceneIndex=sceneIndex;
  const auto decoded=decodeStaticGlb(bytes,options,error);
  if (!decoded) {
    error="Imported Mesh Geometry accepts static meshes in mesh-local space. Animation, skinning and morph extraction is unavailable: "+error;
    return std::nullopt;
  }
  return adaptGlbMeshToGeometryCore(*decoded,meshIndex,valueId,sourceId,revision,error);
}
} // namespace videohelper::gltf
