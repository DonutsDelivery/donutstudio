#pragma once

#include "GeometryCore.h"
#include "Visual3DScene.h"

namespace videowire::geometry {
struct MaterialTableDrawBinding {
  std::uint32_t object = 0;
  StableId sourceMaterial = 0;
  std::size_t instanceIndex = 0;
};
inline bool sameTableMaterial(const HarmonicMIDI::grid::SceneMaterialRecord& a,
                              const HarmonicMIDI::grid::SceneMaterialRecord& b) noexcept {
  return a.baseColor.x==b.baseColor.x && a.baseColor.y==b.baseColor.y && a.baseColor.z==b.baseColor.z
      && a.emissive.x==b.emissive.x && a.emissive.y==b.emissive.y && a.emissive.z==b.emissive.z
      && a.opacity==b.opacity && a.metallic==b.metallic && a.roughness==b.roughness
      && a.alphaMode==b.alphaMode && a.alphaCutoff==b.alphaCutoff && a.doubleSided==b.doubleSided
      && a.baseColorTexture==b.baseColorTexture && a.metallicRoughnessTexture==b.metallicRoughnessTexture
      && a.normalTexture==b.normalTexture && a.occlusionTexture==b.occlusionTexture
      && a.emissiveTexture==b.emissiveTexture && a.normalScale==b.normalScale && a.occlusionStrength==b.occlusionStrength
      && HarmonicMIDI::grid::sameSceneVolume(a.volume,b.volume);
}

inline bool hasMaterialTable(const ValueDescriptor& value, const MaterializedInstancePlan& retained) noexcept {
  const auto found=retained.materialTables.find(value.stableId);
  return found!=retained.materialTables.end() && !found->second.empty();
}

// Resolve face batches and per-instance overrides into the existing native scene
// records. Vertices stay shared. Each face index is stored once regardless of
// instance count, and both native renderers consume the same material records.
inline bool applyGeometryMaterialTable(const ValueDescriptor& value,
    const MaterializedInstancePlan& retained, const std::vector<AttributeData>* meshAttributes,
    HarmonicMIDI::grid::Visual3DScene& scene, std::size_t firstObject, std::size_t objectCount,
    std::string& error,
    const std::map<StableId,HarmonicMIDI::grid::SceneMaterialId>* surfaceIds = nullptr,
    std::vector<MaterialTableDrawBinding>* drawBindings = nullptr,
    const std::map<StableId,HarmonicMIDI::grid::SceneVolume>* slotVolumes = nullptr) {
  using namespace HarmonicMIDI::grid;
  const auto found=retained.materialTables.find(value.stableId);
  const auto* table=found==retained.materialTables.end() ? nullptr : &found->second;
  const auto* faceIndices=meshAttributes ? findNamedAttribute(*meshAttributes,"materialIndex") : nullptr;
  const auto* instanceIndices=value.carrier==CarrierKind::instances3D ? findNamedAttribute(value.attributes,"materialIndex") : nullptr;
  const bool authoredIndices=std::any_of(value.operations.begin(),value.operations.end(),[](const auto& op) {
    return op.code==OperationCode::setMaterialIndex;
  });
  // Imported face indices are preserved asset metadata until a table is bound.
  if ((!table || table->empty()) && !instanceIndices && (!faceIndices || !authoredIndices)) return true;
  if (!table || table->empty()) { error="Material indices require an authored material table"; return false; }
  if (scene.objectCount>scene.objects.size() || scene.materialCount>scene.materials.size()
      || scene.indexCount>scene.indices.size() || objectCount==0 || firstObject>scene.objectCount
      || objectCount>scene.objectCount-firstObject) {
    error="Material table has no bounded native objects"; return false;
  }
  const auto meshObject=scene.objects[firstObject];
  if (meshObject.firstIndex>scene.indexCount || meshObject.indexCount>scene.indexCount-meshObject.firstIndex
      || meshObject.indexCount==0 || meshObject.indexCount%3!=0) {
    error="Material table requires a bounded triangle index range"; return false;
  }
  const auto faceCount=meshObject.indexCount/3;
  const auto validIndices=[&](const AttributeData* attribute,Domain domain,std::size_t count) {
    if (!attribute) return true;
    if (attribute->descriptor.domain!=domain || attribute->descriptor.valueType!=ValueType::integer
        || attribute->elements.size()!=count) return false;
    for (const auto& index : attribute->elements)
      if (!validFieldElement(index,ValueType::integer) || index.components[0]<0 || index.components[0]>31
          || table->count(static_cast<std::uint8_t>(index.components[0]))==0) return false;
    return true;
  };
  if (!validIndices(faceIndices,Domain::face,faceCount)
      || !validIndices(instanceIndices,Domain::instance,objectCount)) {
    error="Material index must be an Integer on the face or instance domain and select an authored slot from zero to 31";
    return false;
  }
  if (!faceIndices && !instanceIndices && table->count(0)==0) {
    error="Material table has no default slot zero"; return false;
  }
  if (value.carrier==CarrierKind::instances3D
      && !validInstanceAppearanceAttributes(value.attributes,objectCount,error)) return false;
  std::map<std::uint8_t,std::vector<std::uint32_t>> groups;
  if (!instanceIndices)
    for (std::size_t face=0;face<faceCount;++face) {
      const auto slot=faceIndices ? static_cast<std::uint8_t>(faceIndices->elements[face].components[0]) : 0;
      auto& indices=groups[slot];
      for (std::size_t corner=0;corner<3;++corner) indices.push_back(scene.indices[meshObject.firstIndex+face*3+corner]);
    }
  std::vector<std::uint32_t> reordered;
  std::map<std::uint8_t,std::pair<std::uint32_t,std::uint32_t>> ranges;
  for (const auto& group : groups) {
    ranges[group.first]={meshObject.firstIndex+static_cast<std::uint32_t>(reordered.size()),static_cast<std::uint32_t>(group.second.size())};
    reordered.insert(reordered.end(),group.second.begin(),group.second.end());
  }
  std::vector<SceneMaterialRecord> materials;
  for (std::size_t material=0;material<scene.materialCount;++material)
    for (std::size_t object=0;object<scene.objectCount;++object)
      if ((object<firstObject || object>=firstObject+objectCount)
          && scene.objects[object].material==scene.materials[material].id) {
        materials.push_back(scene.materials[material]); break;
      }
  std::vector<SceneObjectRecord> objects;
  std::set<std::uint32_t> ids;
  for (std::size_t i=0;i<scene.objectCount;++i) ids.insert(scene.objects[i].id.value);
  // The medium of a slot's source material: a Volume Material, or closed glass
  // and water on a Surface Material.
  const auto slotVolume=[&](StableId source) {
    if (!slotVolumes) return SceneVolume{};
    const auto found=slotVolumes->find(source);
    return found==slotVolumes->end() ? SceneVolume{} : found->second;
  };
  const auto materialFor=[&](std::uint8_t slot,std::size_t instance)->SceneMaterialId {
    const auto& op=table->at(slot); const auto& p=op.field.values;
    if (surfaceIds) {
      const auto surface=surfaceIds->find(op.secondaryInputStableId);
      if (surface!=surfaceIds->end()) {
        const auto existing=std::find_if(materials.begin(),materials.end(),[&](const auto& candidate){return candidate.id==surface->second;});
        if (existing!=materials.end()) return existing->id;
        if (!surface->second.isValid()) return {};
        SceneMaterialRecord material {}; material.id=surface->second;
        material.volume=slotVolume(op.secondaryInputStableId);
        materials.push_back(material); return material.id;
      }
    }
    const auto appearance=value.carrier==CarrierKind::instances3D ? instanceAppearance(value.attributes,instance) : InstanceAppearance{};
    SceneMaterialRecord material {};
    material.baseColor={static_cast<float>(p[0])*appearance.color[0],static_cast<float>(p[1])*appearance.color[1],static_cast<float>(p[2])*appearance.color[2]};
    const auto emission=[&](int i) { return std::min(static_cast<float>(p[3+i])+appearance.emission[i],std::numeric_limits<float>::max()); };
    material.emissive={emission(0),emission(1),emission(2)};
    material.metallic=static_cast<float>(p[6]); material.roughness=static_cast<float>(p[7]);
    if (appearance.metallic>=0) material.metallic=appearance.metallic;
    if (appearance.roughness>=0) material.roughness=appearance.roughness;
    material.opacity=op.parameters[0]*appearance.color[3];
    if (material.opacity<1) material.alphaMode=SceneAlphaMode::Blend;
    material.volume=slotVolume(op.secondaryInputStableId);
    const auto reserved=[&](SceneMaterialId id) {
      return surfaceIds && std::any_of(surfaceIds->begin(),surfaceIds->end(),
          [id](const auto& entry){return entry.second==id;});
    };
    const auto existing=std::find_if(materials.begin(),materials.end(),[&](const auto& candidate){return !reserved(candidate.id) && sameTableMaterial(candidate,material);});
    if (existing!=materials.end()) return existing->id;
    std::uint32_t id=1;
    while (reserved(SceneMaterialId{id}) || std::any_of(materials.begin(),materials.end(),[&](const auto& candidate){return candidate.id.value==id;})) ++id;
    material.id={id}; materials.push_back(material); return material.id;
  };
  for (std::size_t i=0;i<objectCount;++i) {
    const auto source=scene.objects[firstObject+i];
    if (source.firstVertex!=meshObject.firstVertex || source.vertexCount!=meshObject.vertexCount
        || source.firstIndex!=meshObject.firstIndex || source.indexCount!=meshObject.indexCount) {
      error="Material table requires one retained shared mesh"; return false;
    }
    if (instanceIndices) {
      const auto slot=static_cast<std::uint8_t>(instanceIndices->elements[i].components[0]);
      auto draw=source; draw.material=materialFor(slot,i);
      if (!draw.material.isValid()) { error="Material table and instance appearance exceed 32 unique materials"; return false; }
      if (drawBindings) drawBindings->push_back({draw.id.value,table->at(slot).secondaryInputStableId,i});
      objects.push_back(draw); continue;
    }
    if (ranges.size()>1) {
      auto parent=source; parent.material={}; parent.firstVertex=parent.vertexCount=parent.firstIndex=parent.indexCount=0;
      objects.push_back(parent);
    }
    for (const auto& range : ranges) {
      auto draw=source; draw.firstIndex=range.second.first; draw.indexCount=range.second.second;
      draw.material=materialFor(range.first,i);
      if (!draw.material.isValid()) { error="Material table and instance appearance exceed 32 unique materials"; return false; }
      if (ranges.size()>1) {
        std::uint32_t id=0x80000000u | ((source.id.value*16777619u)^static_cast<std::uint32_t>(range.first+1));
        while (!ids.insert(id).second) id=id==std::numeric_limits<std::uint32_t>::max() ? 1 : id+1;
        draw.id={id}; draw.parent=source.id; draw.transform={};
      }
      if (drawBindings) drawBindings->push_back({draw.id.value,table->at(range.first).secondaryInputStableId,i});
      objects.push_back(draw);
    }
  }
  if (objects.size()>scene.objects.size()-(scene.objectCount-objectCount)) {
    error="Material face batches exceed 64 native scene objects"; return false;
  }
  std::vector<SceneObjectRecord> finalObjects(scene.objects.begin(),scene.objects.begin()+firstObject);
  finalObjects.insert(finalObjects.end(),objects.begin(),objects.end());
  finalObjects.insert(finalObjects.end(),scene.objects.begin()+firstObject+objectCount,scene.objects.begin()+scene.objectCount);
  std::copy(finalObjects.begin(),finalObjects.end(),scene.objects.begin()); scene.objectCount=finalObjects.size();
  std::copy(materials.begin(),materials.end(),scene.materials.begin()); scene.materialCount=materials.size();
  if (!reordered.empty()) std::copy(reordered.begin(),reordered.end(),scene.indices.begin()+meshObject.firstIndex);
  error.clear(); return true;
}
} // namespace videowire::geometry
