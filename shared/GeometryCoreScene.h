#pragma once

#include "GeometryCore.h"
#include "Visual3DScene.h"
#include "GeometryMaterialTable.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <new>
#include <string>

namespace videowire::geometry {

inline constexpr const char* kRetainedSceneOperation = "visual.3d.scene.retained";

// Timeline-only composition retains one mesh. Score Sample Fields read the frame's
// canonical score through the composed evaluation. Spectrum bindings read the frame's
// spectrum through the SpectrumFrameBinder: Audio Deformers, in-graph Audio Spectrum
// Fields and Set Position displacements all run inside the replay. Instance transforms
// and Score Field bindings still need their adapters, and material tables can split one
// mesh into draws.
inline bool audioDeformerOnlyBindings(const ValueDescriptor& value) noexcept {
  return !value.spectrumFields.empty()
      && std::all_of(value.spectrumFields.begin(), value.spectrumFields.end(), [](const auto& binding) {
           return binding.target == spectrum::Target::geometryDeformation;
         });
}
inline bool composableSpectrumBindings(const ValueDescriptor& value) noexcept {
  return std::all_of(value.spectrumFields.begin(), value.spectrumFields.end(), [](const auto& binding) {
    return binding.target == spectrum::Target::geometryDeformation
        || binding.target == spectrum::Target::vertexDisplacement;
  });
}
inline bool supportsTimelineSceneGeometry(const ValueDescriptor& value) noexcept {
  return value.carrier == CarrierKind::geometry3D
      && (hasRuntimeFields(value) || !value.spectrumFields.empty())
      && composableSpectrumBindings(value) && value.scoreFields.empty()
      && std::none_of(value.operations.begin(), value.operations.end(), [&](const auto& op) {
          return op.code == OperationCode::materialTableSlot || op.code == OperationCode::setMaterialIndex
              || (op.code == OperationCode::evaluateField && op.inputStableId != op.outputStableId
                  && std::none_of(value.spectrumFields.begin(), value.spectrumFields.end(), [&](const auto& binding) {
                       return binding.fieldStableId == op.stableId;
                     }))
              || (op.retainedMesh && !op.retainedMesh->importedAnimation.empty());
      });
}

// A Spectrum Instancer composes its instances as child objects of one parent. Each frame
// the timeline evaluator rewrites their transforms from the replay and the spectrum.
// Bindings sampled inside the replay by an Audio Spectrum Field operation stay with the
// image path's ordered instancing.
inline bool supportsTimelineSceneInstances(const ValueDescriptor& value) noexcept {
  return value.carrier == CarrierKind::instances3D && !value.spectrumFields.empty()
      && value.scoreFields.empty() && !hasRuntimeFields(value)
      && std::all_of(value.spectrumFields.begin(), value.spectrumFields.end(), [&](const auto& binding) {
           return binding.target == spectrum::Target::instanceTransform
               && std::none_of(value.operations.begin(), value.operations.end(), [&](const auto& op) {
                    return op.code == OperationCode::fieldAudioSpectrum && op.stableId == binding.fieldStableId;
                  });
         })
      && std::none_of(value.operations.begin(), value.operations.end(), [](const auto& op) {
          return op.code == OperationCode::materialTableSlot || op.code == OperationCode::setMaterialIndex
              || (op.retainedMesh && !op.retainedMesh->importedAnimation.empty());
      });
}

// Sets one frame's spectrum readers on a composed replay's evaluation for this value.
using SpectrumFrameBinder = std::function<bool(RuntimeFieldEvaluation&, const ValueDescriptor&, std::string&)>;

// Both image rendering and composed scenes select the terminal mesh from the
// same ordered Geometry Core replay, including its evaluated vertex attributes.
inline const GeometryData* replayGeometryMesh(const ValueDescriptor& value,
    MaterializedInstancePlan& retained, std::string& error,
    const RuntimeFieldEvaluation* runtime = nullptr,
    const std::function<bool(GeometryData&, const spectrum::Binding&, std::string&)>& deform = {}) {
  const auto& frameDeform = deform || !runtime ? deform : runtime->audioDeformAt;
  if (!validateOperationPlan(value, {}, &retained, error, nullptr, runtime, frameDeform)) return nullptr;
  if (value.carrier != CarrierKind::geometry3D) return nullptr;
  if (runtime || frameDeform) {
    const auto terminal = retained.geometries.find(value.stableId);
    if (terminal == retained.geometries.end()) {
      error = "Reactive geometry replay requires a retained terminal mesh; this operation combination is unavailable";
      return nullptr;
    }
    return &terminal->second;
  }
  return &std::get<GeometryData>(value.data);
}

// Compact an existing Geometry Core identity into the native scene's object-id
// range. Keep the graph object as a transform parent, and reject collisions.
inline HarmonicMIDI::grid::SceneObjectId sceneInstanceId(
    HarmonicMIDI::grid::SceneObjectId object, StableId instance) noexcept {
  std::uint64_t hash = 14695981039346656037ull;
  for (const auto value : {static_cast<std::uint64_t>(object.value), instance})
    for (unsigned byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8u)) & 0xffu;
      hash *= 1099511628211ull;
    }
  return {0x80000000u | static_cast<std::uint32_t>(hash & 0x7fffffffu)};
}

// Reference viewport height for dicing in a composed scene. The composition has
// no output size of its own, so Subdivision Surface rates count pixels of a
// 1080-line frame.
inline constexpr double kComposedSceneDicingHeight = 1080.0;

// Row-major 3x4 transform from an object's local space (its parent chain
// included) to the space of the camera, which sits at the origin. The camera
// record is camera-to-world with no scale, so world to camera is R^T (p - t).
inline std::array<double,12> sceneLocalToCamera(const HarmonicMIDI::grid::Visual3DScene& scene,
    const HarmonicMIDI::grid::SceneObjectRecord& object,
    const HarmonicMIDI::grid::SceneCameraRecord& camera) noexcept {
  using Matrix = std::array<double,12>;
  const auto rotation = [](const HarmonicMIDI::grid::SceneQuaternion& q) {
    const double x=q.x, y=q.y, z=q.z, w=q.w, n=std::sqrt(x*x+y*y+z*z+w*w);
    const double a=n>0 ? x/n : 0, b=n>0 ? y/n : 0, c=n>0 ? z/n : 0, d=n>0 ? w/n : 1;
    return std::array<double,9>{{1-2*(b*b+c*c), 2*(a*b-d*c), 2*(a*c+d*b),
                                 2*(a*b+d*c), 1-2*(a*a+c*c), 2*(b*c-d*a),
                                 2*(a*c-d*b), 2*(b*c+d*a), 1-2*(a*a+b*b)}};
  };
  const auto trs = [&](const HarmonicMIDI::grid::SceneTransform3D& t) {
    const auto r = rotation(t.rotation);
    const double s[3] = {t.scale.x, t.scale.y, t.scale.z}, o[3] = {t.translation.x, t.translation.y, t.translation.z};
    Matrix m{};
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) m[row*4+column] = r[row*3+column]*s[column];
      m[row*4+3] = o[row];
    }
    return m;
  };
  const auto multiply = [](const Matrix& l, const Matrix& r) {
    Matrix m{};
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 4; ++column) {
        double sum = column == 3 ? l[row*4+3] : 0.0;
        for (int inner = 0; inner < 3; ++inner) sum += l[row*4+inner]*r[inner*4+column];
        m[row*4+column] = sum;
      }
    return m;
  };
  auto world = trs(object.transform);
  auto parent = object.parent;
  for (std::size_t depth = 0; parent.isValid() && depth < scene.objectCount; ++depth) {
    const auto* found = HarmonicMIDI::grid::visual3d_detail::findById(scene.objects, scene.objectCount, parent);
    if (!found) break;
    world = multiply(trs(found->transform), world);
    parent = found->parent;
  }
  const auto r = rotation(camera.transform.rotation);
  const double t[3] = {camera.transform.translation.x, camera.transform.translation.y, camera.transform.translation.z};
  Matrix view{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) view[row*4+column] = r[column*3+row];
    view[row*4+3] = -(r[0*3+row]*t[0] + r[1*3+row]*t[1] + r[2*3+row]*t[2]);
  }
  return multiply(view, world);
}

// Append retained geometry without expanding instances into duplicate vertices.
// The caller supplies the scene's camera/light state and validates the final scene.
inline bool appendGeometryToScene(const AdmittedValue& value,
    HarmonicMIDI::grid::SceneObjectRecord object,
    const HarmonicMIDI::grid::SceneMaterialRecord& material,
    HarmonicMIDI::grid::Visual3DScene& scene, std::string& error,
    const std::map<StableId,HarmonicMIDI::grid::SceneMaterialId>* surfaceIds = nullptr,
    std::vector<MaterialTableDrawBinding>* drawBindings = nullptr, bool defaultSurface = false,
    const RuntimeFieldEvaluation* runtime = nullptr,
    const std::map<StableId,HarmonicMIDI::grid::SceneVolume>* slotVolumes = nullptr) {
  using namespace HarmonicMIDI::grid;
  const auto& descriptor = value.descriptor();
  // An Audio Deformer composes at rest without a frame hook; the timeline evaluator
  // replaces its vertices each frame.
  if ((!descriptor.spectrumFields.empty() && !supportsTimelineSceneGeometry(descriptor)
       && !supportsTimelineSceneInstances(descriptor))
      || !descriptor.scoreFields.empty()
      || (hasRuntimeFields(descriptor) && !supportsTimelineSceneGeometry(descriptor))
      || std::any_of(descriptor.operations.begin(),descriptor.operations.end(),[](const auto& operation) {
          return operation.retainedMesh && !operation.retainedMesh->importedAnimation.empty();
      })) {
    error = "Scene composition does not yet evaluate reactive Geometry Core bindings; use Geometry Scene for this source";
    return false;
  }
  // Subdivision Surface dicing replays the static mesh from the scene's active
  // camera, the same replay the native frame runs. Timeline geometry keeps its
  // authored level: its frame rebinds need one topology.
  RuntimeFieldEvaluation dicingView;
  if (!runtime && descriptor.carrier == CarrierKind::geometry3D && !hasRuntimeFields(descriptor)
      && hasAdaptiveSubdivision(descriptor)) {
    const auto* camera = HarmonicMIDI::grid::visual3d_detail::findById(scene.cameras,
        std::min(scene.cameraCount, scene.cameras.size()), scene.activeCamera);
    if (!camera && scene.cameraCount > 0) camera = &scene.cameras[0];
    if (camera) {
      dicingView.subdivisionLocalToCamera = sceneLocalToCamera(scene, object, *camera);
      dicingView.subdivisionTanHalfFov = std::tan(static_cast<double>(camera->verticalFovRadians) / 2.0);
      dicingView.subdivisionViewportHeight = kComposedSceneDicingHeight;
      runtime = &dicingView;
    }
  }
  MaterializedInstancePlan retained;
  const GeometryData* mesh = nullptr;
  if (descriptor.carrier == CarrierKind::geometry3D) {
    mesh = replayGeometryMesh(descriptor, retained, error, runtime);
    if (!mesh) return false;
  } else if (!validateOperationPlan(descriptor, {}, &retained, error)) return false;
  const std::vector<AttributeData>* attributes=&descriptor.attributes;
  const InstancesData* instances = nullptr;
  if (descriptor.carrier == CarrierKind::geometry3D) {
    if (runtime) attributes = &retained.terminalAttributes;
  }
  else if (descriptor.carrier == CarrierKind::instances3D) {
    instances = &retained.instances;
    if (instances->instances.empty()) {
      error = "Scene composition requires at least one retained instance";
      return false;
    }
    const auto sourceId = instances->instances.front().sourceStableId;
    const auto source = retained.geometries.find(sourceId);
    if (source == retained.geometries.end()
        || !std::all_of(instances->instances.begin(), instances->instances.end(),
             [sourceId](const auto& item) { return item.sourceStableId == sourceId; })) {
      error = "Scene composition requires one retained mesh source per instance set";
      return false;
    }
    mesh = &source->second;
    const auto found=retained.geometryAttributes.find(sourceId);
    attributes=found==retained.geometryAttributes.end() ? nullptr : &found->second;
  } else {
    error = "Scene composition requires Geometry3D or Instances3D";
    return false;
  }
  std::vector<SceneMaterialRecord> appearanceMaterials;
  std::vector<SceneMaterialId> instanceMaterials;
  const bool table=hasMaterialTable(descriptor,retained);
  const std::size_t objectsAdded=instances ? instances->instances.size()+1 : 1;
  if (instances != nullptr && !table && !defaultSurface) {
    std::vector<InstanceAppearance> variants;
    std::set<std::uint32_t> materialIds{material.id.value};
    for (std::size_t i=0;i<scene.materialCount;++i) materialIds.insert(scene.materials[i].id.value);
    for (std::size_t i=0;i<instances->instances.size();++i) {
      const auto appearance=instanceAppearance(descriptor.attributes,i);
      if (appearance.color==InstanceAppearance{}.color && appearance.emission==InstanceAppearance{}.emission
          && appearance.metallic<0 && appearance.roughness<0) {
        instanceMaterials.push_back(material.id); continue;
      }
      const auto existing=std::find_if(variants.begin(),variants.end(),[&](const auto& candidate) {
        return candidate.color==appearance.color && candidate.emission==appearance.emission
            && candidate.metallic==appearance.metallic && candidate.roughness==appearance.roughness;
      });
      if (existing!=variants.end()) {
        instanceMaterials.push_back(appearanceMaterials[static_cast<std::size_t>(existing-variants.begin())].id);
        continue;
      }
      auto variant=material;
      auto id=sceneInstanceId(object.id,instances->instances[i].stableId).value;
      while (!materialIds.insert(id).second) id=id==std::numeric_limits<std::uint32_t>::max() ? 1 : id+1;
      variant.id={id};
      variant.baseColor={material.baseColor.x*appearance.color[0],material.baseColor.y*appearance.color[1],
                         material.baseColor.z*appearance.color[2]};
      variant.emissive={std::min(material.emissive.x+appearance.emission[0],std::numeric_limits<float>::max()),
                        std::min(material.emissive.y+appearance.emission[1],std::numeric_limits<float>::max()),
                        std::min(material.emissive.z+appearance.emission[2],std::numeric_limits<float>::max())};
      variant.opacity*=appearance.color[3];
      if (appearance.metallic>=0) variant.metallic=appearance.metallic;
      if (appearance.roughness>=0) variant.roughness=appearance.roughness;
      if (variant.opacity<1) variant.alphaMode=SceneAlphaMode::Blend;
      instanceMaterials.push_back(variant.id);
      variants.push_back(appearance); appearanceMaterials.push_back(variant);
    }
  }
  if (mesh->positions.empty() || mesh->indices.empty()) {
    error = "Scene composition requires a mesh with vertices and indices";
    return false;
  }
  Visual3DSceneGrowth added;
  added.vertices = mesh->positions.size();
  added.indices = mesh->indices.size();
  added.objects = objectsAdded;
  added.materials = (table ? std::size_t{0} : std::size_t{1}) + appearanceMaterials.size();
  std::set<std::uint32_t> ids;
  for (std::size_t index = 0; index < scene.objectCount; ++index)
    ids.insert(scene.objects[index].id.value);
  if (!object.id.isValid() || !material.id.isValid() || !ids.insert(object.id.value).second) {
    error = "Scene composition has an invalid or duplicate object identity";
    return false;
  }
  if (instances != nullptr)
    for (const auto& instance : instances->instances)
      if (!ids.insert(sceneInstanceId(object.id, instance.stableId).value).second) {
        error = "Scene composition cannot represent these instance identities without a collision";
        return false;
      }
  for (const auto index : mesh->indices)
    if (index >= mesh->positions.size()) {
      error = "Scene composition has an out-of-range geometry index";
      return false;
    }
  if (!growVisual3DScene(scene, added, error)) return false;
  const auto vertexBase = scene.vertexCount;
  for (std::size_t index = 0; index < mesh->positions.size(); ++index) {
    const auto& p = mesh->positions[index];
    scene.vertices[vertexBase + index] = {};
    scene.vertices[vertexBase + index].position = {p.x, p.y, p.z};
  }
  for (std::size_t index = 0; index < mesh->indices.size(); ++index)
    scene.indices[scene.indexCount + index] = mesh->indices[index];
  // A vertex normal attribute of the mesh's cardinality overwrites every face normal
  // below or refuses the scene, so the face sums are only built without one.
  const bool attributeNormals = attributes && std::any_of(attributes->begin(), attributes->end(),
      [&](const AttributeData& attribute) {
        return attribute.descriptor.name == "normal" && attribute.descriptor.domain == Domain::vertex
            && attribute.descriptor.valueType == ValueType::vector
            && attribute.elements.size() == mesh->positions.size();
      });
  for (std::size_t index = 0; !attributeNormals && index + 2 < mesh->indices.size(); index += 3) {
    const auto a = mesh->indices[index], b = mesh->indices[index + 1], c = mesh->indices[index + 2];
    const auto& p = mesh->positions[a];
    const auto& q = mesh->positions[b];
    const auto& r = mesh->positions[c];
    const Vec3 normal {(q.y-p.y)*(r.z-p.z)-(q.z-p.z)*(r.y-p.y),
                       (q.z-p.z)*(r.x-p.x)-(q.x-p.x)*(r.z-p.z),
                       (q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x)};
    for (const auto vertex : {a,b,c}) {
      auto& n = scene.vertices[vertexBase + vertex].normal;
      n.x += normal.x; n.y += normal.y; n.z += normal.z;
    }
  }
  for (std::size_t index = 0; !attributeNormals && index < mesh->positions.size(); ++index) {
    auto& n = scene.vertices[vertexBase + index].normal;
    const auto length = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    if (length > 0) { n.x /= length; n.y /= length; n.z /= length; }
    else n = {0,0,1};
  }
  object.material = material.id;
  // Generator meshes that carry their own uv (plane, sphere, cylinder)
  // publish it here.
  // attributes is a pointer at the descriptor's vector for geometry3D, so it
  // is never null even when empty; the publish is gated on the ABSENCE of an
  // authored uv attribute instead. A retained mesh with an authored uv
  // attribute still wins, because the loop below overwrites these values.
  const bool authoredUv = attributes && std::any_of(attributes->begin(), attributes->end(),
      [] (const AttributeData& attribute) { return attribute.descriptor.name == "uv"; });
  if (!authoredUv && mesh->uv.size()==mesh->positions.size())
    for (std::size_t index=0;index<mesh->uv.size();++index)
      scene.vertices[vertexBase+index].uv={mesh->uv[index].x, mesh->uv[index].y};
  if (attributes!=nullptr)
    for (const auto& attribute : *attributes) {
      const auto& name=attribute.descriptor.name;
      // A vertex Float "roughness" (Ocean foam through Store Attribute, say)
      // fills the vertex roughness lane at full weight, clamped to [0, 1].
      if (name=="roughness" && attribute.descriptor.domain==Domain::vertex) {
        if (attribute.descriptor.valueType!=ValueType::floatValue || attribute.elements.size()!=mesh->positions.size()) {
          error="Scene vertex roughness must be one Float per vertex"; return false;
        }
        for (std::size_t index=0;index<attribute.elements.size();++index) {
          const auto value=attribute.elements[index].components[0];
          if (!std::isfinite(value)) { error="Scene vertex roughness is not finite"; return false; }
          scene.vertices[vertexBase+index].roughness={static_cast<float>(std::clamp(value,0.0,1.0)),1.0f};
        }
        continue;
      }
      if (name!="normal" && name!="uv" && name!="color") continue;
      if (attribute.descriptor.domain!=Domain::vertex || attribute.elements.size()!=mesh->positions.size()
          || attribute.descriptor.valueType!=(name=="color" ? ValueType::color : ValueType::vector)) {
        error="Scene vertex attribute has the wrong type or cardinality"; return false;
      }
      for (std::size_t index=0;index<attribute.elements.size();++index) {
        auto& vertex=scene.vertices[vertexBase+index]; const auto& value=attribute.elements[index].components;
        if (name=="uv") vertex.uv={static_cast<float>(value[0]),static_cast<float>(value[1])};
        else if (name=="color") vertex.color={static_cast<float>(value[0]),static_cast<float>(value[1]),
            static_cast<float>(value[2]),static_cast<float>(value[3])};
        else {
          const auto length=std::sqrt(value[0]*value[0]+value[1]*value[1]+value[2]*value[2]);
          if (!std::isfinite(length) || length==0) { error="Scene vertex normal is invalid"; return false; }
          vertex.normal={static_cast<float>(value[0]/length),static_cast<float>(value[1]/length),static_cast<float>(value[2]/length)};
        }
      }
    }
  object.authoredShading = std::any_of(descriptor.operations.begin(),descriptor.operations.end(),
      [](const auto& operation) { return operation.code==OperationCode::shadeMesh; });
  object.firstVertex = static_cast<std::uint32_t>(scene.vertexCount);
  object.vertexCount = static_cast<std::uint32_t>(mesh->positions.size());
  object.firstIndex = static_cast<std::uint32_t>(scene.indexCount);
  object.indexCount = static_cast<std::uint32_t>(mesh->indices.size());
  scene.vertexCount += mesh->positions.size();
  scene.indexCount += mesh->indices.size();
  if (!table) scene.materials[scene.materialCount++] = material;
  for (const auto& variant : appearanceMaterials) scene.materials[scene.materialCount++]=variant;
  if (instances == nullptr) scene.objects[scene.objectCount++] = object;
  else {
    auto parent = object;
    parent.material = {};
    parent.firstVertex = parent.vertexCount = parent.firstIndex = parent.indexCount = 0;
    scene.objects[scene.objectCount++] = parent;
    for (std::size_t index=0;index<instances->instances.size();++index) {
      const auto& instance=instances->instances[index];
      auto child = object;
      child.material=table || defaultSurface ? material.id : instanceMaterials[index];
      child.id = sceneInstanceId(object.id, instance.stableId);
      child.parent = object.id;
      child.transform.translation = {instance.transform.translation.x, instance.transform.translation.y,
                                     instance.transform.translation.z};
      child.transform.rotation = {instance.transform.rotation.x, instance.transform.rotation.y,
                                  instance.transform.rotation.z, instance.transform.rotation.w};
      child.transform.scale = {instance.transform.scale.x, instance.transform.scale.y, instance.transform.scale.z};
      scene.objects[scene.objectCount++] = child;
    }
  }
  const auto drawCount=instances ? instances->instances.size() : std::size_t{1};
  const auto firstDraw=scene.objectCount-drawCount;
  if (!applyGeometryMaterialTable(descriptor,retained,attributes,scene,firstDraw,drawCount,error,surfaceIds,drawBindings,slotVolumes)) return false;
  if (!table && drawBindings)
    for (std::size_t index=0;index<drawCount;++index)
      drawBindings->push_back({scene.objects[firstDraw+index].id.value,0,index});
  error.clear();
  return true;
}
} // namespace videowire::geometry
