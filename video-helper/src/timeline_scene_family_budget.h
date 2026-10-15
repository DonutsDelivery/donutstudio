#pragma once

#include "../../shared/GeometryTimelineSceneBinding.h"

#include <cstring>
#include <map>
#include <tuple>

namespace videohelper::geometry::timelinefamily {
using Scene = HarmonicMIDI::grid::Visual3DScene;
using namespace videowire::geometry;

// Logical application storage and counted visits, not allocator or driver memory.
struct Envelope final {
  std::size_t retainedBytes = 0;
  std::size_t attemptBytes = 0;
  std::size_t decodeByteVisits = 0;
  std::size_t rangeComparisons = 0;
  std::size_t replayIterations = 0; // Three future frame replays, at most 384.
  std::size_t admissionReplayIterations = 0;
  std::size_t planPairVisits = 0;
  // Entry reads only sizes/capacities, then checked arithmetic. No scene walk.
  std::size_t entryCountChecks = 0;
  std::size_t entryRecordVisits = 0;
  std::size_t residentOwnerVisits = 0;
  std::size_t sceneValidationCalls = 0;
  std::size_t hierarchyLookupVisits = 0;
  std::size_t hierarchyDepthIterations = 0;
  std::size_t matrixConstructions = 0;
  std::size_t matrixMultiplyTerms = 0;
  std::size_t matrixElementVisits = 0;
  std::size_t matrixPostcheckIterations = 0;
  std::size_t lightChannelVisits = 0;
  // A record visit covers its fixed straight-line predicate or comparison,
  // not a scalar-operation count. Searches and nested loops count each visit.
  std::size_t sceneRecordVisits = 0;
  std::size_t sceneStringByteVisits = 0;
  std::size_t fieldElementVisits = 0;
  std::size_t normalIndexVisits = 0;
  std::size_t waterSampleElements = 0;
  std::size_t sceneCopyElements = 0;
};

inline bool sameVector(const HarmonicMIDI::grid::SceneVec3& a,
                       const HarmonicMIDI::grid::SceneVec3& b) noexcept {
  return std::memcmp(&a.x,&b.x,sizeof(float))==0
      && std::memcmp(&a.y,&b.y,sizeof(float))==0
      && std::memcmp(&a.z,&b.z,sizeof(float))==0;
}
inline bool sameFloat(float a,float b) noexcept {
  return std::memcmp(&a,&b,sizeof(float))==0;
}
inline bool sameTransform(const HarmonicMIDI::grid::SceneTransform3D& a,
                          const HarmonicMIDI::grid::SceneTransform3D& b) noexcept {
  return sameVector(a.translation,b.translation) && sameVector(a.scale,b.scale)
      && sameFloat(a.rotation.x,b.rotation.x) && sameFloat(a.rotation.y,b.rotation.y)
      && sameFloat(a.rotation.z,b.rotation.z) && sameFloat(a.rotation.w,b.rotation.w);
}
inline bool sameScene(const Scene& a,const Scene& b) noexcept {
  using namespace HarmonicMIDI::grid;
  if (a.schemaVersion!=b.schemaVersion || a.id!=b.id || a.activeCamera!=b.activeCamera
      || !sameVector(a.ambientColor,b.ambientColor)
      || a.vertexCount!=b.vertexCount || a.indexCount!=b.indexCount
      || a.objectCount!=b.objectCount || a.materialCount!=b.materialCount
      || a.textureCount!=b.textureCount || a.textureTexelCount!=b.textureTexelCount
      || a.lightCount!=b.lightCount || a.cameraCount!=b.cameraCount
      || a.textureTexels.size()!=b.textureTexels.size()) return false;
  // Compare declared fields, including unused fixed records, without padding.
  for (std::size_t i=0;i<a.vertices.size();++i) {
    const auto& x=a.vertices[i]; const auto& y=b.vertices[i];
    if (!sameVector(x.position,y.position) || !sameVector(x.normal,y.normal)
        || !sameFloat(x.uv.x,y.uv.x) || !sameFloat(x.uv.y,y.uv.y)
        || std::memcmp(x.color.data(),y.color.data(),sizeof(x.color))!=0
        || std::memcmp(x.tangent.data(),y.tangent.data(),sizeof(x.tangent))!=0) return false;
  }
  if (a.indices!=b.indices) return false;
  for (std::size_t i=0;i<a.objects.size();++i) {
    const auto& x=a.objects[i]; const auto& y=b.objects[i];
    if (std::tie(x.id,x.parent,x.material,x.firstVertex,x.vertexCount,x.firstIndex,x.indexCount)
        != std::tie(y.id,y.parent,y.material,y.firstVertex,y.vertexCount,y.firstIndex,y.indexCount)
        || !sameTransform(x.transform,y.transform)) return false;
  }
  for (std::size_t i=0;i<a.materials.size();++i) {
    const auto& x=a.materials[i]; const auto& y=b.materials[i];
    if (x.id!=y.id || x.baseColorTexture!=y.baseColorTexture
        || x.metallicRoughnessTexture!=y.metallicRoughnessTexture || x.normalTexture!=y.normalTexture
        || x.occlusionTexture!=y.occlusionTexture || x.emissiveTexture!=y.emissiveTexture
        || x.alphaMode!=y.alphaMode || x.doubleSided!=y.doubleSided
        || !sameVector(x.baseColor,y.baseColor) || !sameVector(x.emissive,y.emissive)
        || !sameFloat(x.opacity,y.opacity) || !sameFloat(x.metallic,y.metallic)
        || !sameFloat(x.roughness,y.roughness) || !sameFloat(x.normalScale,y.normalScale)
        || !sameFloat(x.occlusionStrength,y.occlusionStrength) || !sameFloat(x.alphaCutoff,y.alphaCutoff)) return false;
  }
  for (std::size_t i=0;i<a.textures.size();++i) {
    const auto& x=a.textures[i]; const auto& y=b.textures[i];
    if (std::tie(x.id,x.firstTexel,x.width,x.height,x.magFilter,x.minFilter,x.wrapS,x.wrapT)
        != std::tie(y.id,y.firstTexel,y.width,y.height,y.magFilter,y.minFilter,y.wrapS,y.wrapT)) return false;
  }
  for (std::size_t i=0;i<a.textureTexels.size();++i) {
    const auto& x=a.textureTexels[i]; const auto& y=b.textureTexels[i];
    if (std::tie(x.red,x.green,x.blue,x.alpha)!=std::tie(y.red,y.green,y.blue,y.alpha)) return false;
  }
  for (std::size_t i=0;i<a.lights.size();++i) {
    const auto& x=a.lights[i]; const auto& y=b.lights[i];
    if (x.id!=y.id || x.kind!=y.kind || !sameTransform(x.transform,y.transform)
        || !sameVector(x.color,y.color) || !sameFloat(x.intensity,y.intensity)
        || !sameFloat(x.range,y.range) || !sameFloat(x.innerConeAngle,y.innerConeAngle)
        || !sameFloat(x.outerConeAngle,y.outerConeAngle) || !sameAnalyticSky(x.sky,y.sky)
        || !sameEnvironmentImage(x.environmentImage,y.environmentImage)
        || !sameSpectralSource(x.spectralSource,y.spectralSource)) return false;
  }
  for (std::size_t i=0;i<a.cameras.size();++i) {
    const auto& x=a.cameras[i]; const auto& y=b.cameras[i];
    if (x.id!=y.id || !sameTransform(x.transform,y.transform)
        || !sameFloat(x.verticalFovRadians,y.verticalFovRadians)
        || !sameFloat(x.nearPlane,y.nearPlane) || !sameFloat(x.farPlane,y.farPlane)) return false;
  }
  return true;
}

// Source schedule for validateVisual3DScene, sameScene and the binding joins.
// Use fixed capacities even for sparse scenes: replay/default construction can
// visit records absent from the authored counts. Early returns only reduce work.
inline bool deriveSceneWork(std::size_t children,std::size_t texels,
    std::size_t environmentBytes,std::size_t text,std::size_t childRecordBytes,
    Envelope& out) noexcept {
  constexpr auto O=Scene::kMaxObjects, M=Scene::kMaxMaterials, T=Scene::kMaxTextures;
  constexpr auto L=Scene::kMaxLights, C=Scene::kMaxCameras;
  constexpr auto V=Scene::kMaxVertices, I=Scene::kMaxIndices;
  constexpr auto S=diffractionmaterial::kMaximumSpectralSamples;
  std::size_t calls=0,perValidation=0,searches=0,depths=0,multiplies=0;
  std::size_t completeRecords=0,bindingVisits=0,copies=0,stringPasses=0,terms=0;
  // Initial admission, one validTimelineSceneBinding per admitted child, then
  // two validations per future replacement. Cache reuse needs only the first.
  if (!checkedMul(children,3,calls) || !checkedAdd(calls,1,calls)
      || !checkedMul(O,O+1,depths) || !checkedMul(O,O,multiplies)
      || !checkedAdd(depths,multiplies,terms)
      || !checkedMul(terms,O,searches) || !checkedMul(calls,searches,out.hierarchyLookupVisits)
      || !checkedMul(calls,terms,out.hierarchyDepthIterations)
      || !checkedAdd(multiplies,O,terms) || !checkedMul(calls,terms,out.matrixConstructions)
      || !checkedMul(calls,multiplies,terms) || !checkedMul(terms,4*4*4,out.matrixMultiplyTerms)
      // Each matrixFor initializes and transfers 16 elements. Each multiply
      // zeroes 16 and assigns 16 world elements. Postchecks read 9 scale, 18 dot and 3
      // translation elements. Each of the 64 terms reads two and updates one
      // result element. Arithmetic inside each term is fixed.
      || !checkedMul(out.matrixConstructions,16+16,out.matrixElementVisits)
      || !addArrayBytes(terms,16+16+3*4*4*4,out.matrixElementVisits)
      || !addArrayBytes(calls,O*(9+18+3),out.matrixElementVisits)
      || !checkedMul(calls,O*(3+3),out.matrixPostcheckIterations)) return false;
  out.sceneValidationCalls=calls;
  // All duplicateIdAt earlier-record scans, parent-presence lookup, both
  // hierarchy searches, object material lookup, five material texture lookups
  // and the active-camera lookup. Object index ranges can overlap, so validation
  // must reserve O*I index checks rather than assume one pass over indices.
  for (const auto count : {O,M,T,L,C})
    if (!addArrayBytes(count,count-1,perValidation)) return false;
  perValidation/=2;
  if (!checkedAdd(perValidation,searches,perValidation)
      || !addArrayBytes(O,O+M+I,perValidation)
      || !addArrayBytes(M,5*T,perValidation)
      || !checkedAdd(perValidation,C,perValidation)
      || !checkedAdd(perValidation,1+T+M+2*O+V+2*L+C,perValidation)
      // Three sky channel checks; two spectral arrays and directional
      // coefficients have S entries each, including version-zero comparisons.
      || !checkedAdd(calls,1,terms)
      || !checkedMul(terms,L*(3+3*S),out.lightChannelVisits)
      || !checkedMul(calls,perValidation,out.sceneRecordVisits)
      || !checkedAdd(V+I+O+M+T+L+C,texels,completeRecords)
      // One exact full-scene cache comparison. Three binding range scans:
      // initial admission, admitted baseline and future replacement baseline.
      || !checkedMul(children,3*O,out.rangeComparisons)
      // Five object searches: initial/range, baseline, retained child, future
      // baseline and future replacement. Three material searches accompany the
      // admission baseline and both future baseline/replacement joins. Three
      // binding-record passes cover canonical order, exact plan equality and
      // backend equality. Each fixed scene header is one record visit.
      || !checkedMul(children,5*O+3*M+2*V+3*I+3,bindingVisits)
      || !checkedAdd(out.sceneRecordVisits,completeRecords+1,out.sceneRecordVisits)
      || !checkedAdd(out.sceneRecordVisits,out.rangeComparisons,out.sceneRecordVisits)
      || !checkedAdd(out.sceneRecordVisits,bindingVisits,out.sceneRecordVisits)
      // Frozen copy, future composed-scene copy, and three default candidate
      // scenes per child. Include every fixed record and texture element.
      || !checkedMul(children,3,copies) || !checkedAdd(copies,2,copies)
      || !checkedMul(copies,completeRecords,out.sceneCopyElements)
      // Admission/future baseline append and future evaluated append each
      // write vertices and indices, then replacement writes vertices once.
      || !addArrayBytes(children,4*V+3*I,out.sceneCopyElements)
      // Validator strings, full cache equality, scene copies, binding equality
      // and retention. Backend identity comparisons are bounded by child bytes.
      || !checkedAdd(calls,copies,stringPasses) || !checkedAdd(stringPasses,1,stringPasses)
      || !checkedMul(stringPasses,environmentBytes,out.sceneStringByteVisits)
      || !addArrayBytes(text,2,out.sceneStringByteVisits)
      || !addArrayBytes(children,childRecordBytes,out.sceneStringByteVisits)) return false;
  return true;
}

// Reserve the worst admitted mesh/field per operation before parsing. Serialized
// payload size bounds source arrays and strings; retained sources cannot exceed
// that payload. Replay maps may retain a mesh, attributes and a field per op.
inline bool deriveEnvelope(const Scene& scene,const std::vector<TimelineSceneBinding>& bindings,
    const ResourceLimits& limits,std::size_t childRecordBytes,Envelope& out,std::string& error) {
  std::size_t text=0,binary=0,operations=0,environmentBytes=0;
  // Entry schedule: at most 13 count predicates, 64 binding size/identity
  // records and 16 light string-capacity records. Contents are untouched.
  if (bindings.empty() || bindings.size()>Scene::kMaxObjects
      || scene.vertexCount>Scene::kMaxVertices || scene.indexCount>Scene::kMaxIndices
      || scene.objectCount>Scene::kMaxObjects || scene.materialCount>Scene::kMaxMaterials
      || scene.textureCount>Scene::kMaxTextures || scene.lightCount>Scene::kMaxLights
      || scene.cameraCount>Scene::kMaxCameras || scene.textureTexelCount>Scene::kMaxTextureTexels
      || scene.textureTexelCount!=scene.textureTexels.size()
      || scene.vertexCount>limits.maxVertices || scene.indexCount>limits.maxIndices) {
    error="Timeline scene family exceeds scene counts or 1..64 binding capacity"; return false;
  }
  for (const auto& binding : bindings) {
    const auto n=binding.loweredPlan.size();
    if (!binding.object.isValid() || n==0 || n%2!=0
        || !checkedAdd(text,n,text) || text>kMaximumTimelineScenePlanBytes
        || !checkedAdd(binary,n/2,binary) || binary>kMaximumTimelineScenePlanBytes/2) {
      error="Timeline scene family exceeds aggregate lowered text or binary capacity"; return false;
    }
    // Every operation has at least 41 wire bytes. This upper bound does not
    // inspect untrusted operation records or allocate decoder staging.
    if (!checkedAdd(operations,n/2/41,operations)) return false;
  }
  operations=std::min(operations,std::min<std::size_t>(128,limits.maxOperations));
  if (operations<bindings.size()) { error="Timeline scene family has no remaining operation budget"; return false; }
  for (const auto& light : scene.lights) {
    if (!checkedAdd(environmentBytes,light.environmentImage.asset.contentSha256.capacity(),environmentBytes)
        || !checkedAdd(environmentBytes,light.environmentImage.asset.id.capacity(),environmentBytes)
        || !checkedAdd(environmentBytes,2,environmentBytes)) {
      error="Timeline scene family environment storage overflow"; return false;
    }
  }
  std::size_t sceneBytes=sizeof(Scene),descriptor=sizeof(ValueDescriptor)+sizeof(RetainedMeshData);
  std::size_t mesh=sizeof(GeometryData),attributes=0,field=sizeof(FieldData),replay=0,sources=0;
  std::size_t descriptorCopies=0,replayCopies=0,waterElements=0,dependencyElements=0;
  std::size_t childBytes=0,descriptorBytes=0,waterBytes=0;
  const auto vertices=std::min<std::size_t>(Scene::kMaxVertices,limits.maxVertices);
  const auto indices=std::min<std::size_t>(Scene::kMaxIndices,limits.maxIndices);
  const auto elements=std::min(Scene::kMaxVertices,limits.maxFieldElements);
  const auto attributeCount=std::min(limits.maxAttributes,binary/22);
  if (!checkedAdd(childRecordBytes,sizeof(TimelineSceneBinding),childBytes)
      || !checkedMul(bindings.size(),2,descriptorCopies)
      || !checkedMul(operations,4,replayCopies)
      || !checkedMul(operations,vertices,waterElements)
      || !checkedMul(operations,operations,dependencyElements)
      || !addArrayBytes(scene.textureTexels.capacity(),sizeof(HarmonicMIDI::grid::SceneTexelRgba8),sceneBytes)
      || !checkedAdd(sceneBytes,environmentBytes,sceneBytes)
      || !addArrayBytes(vertices,sizeof(Vec3)+sizeof(StableId),mesh)
      || !addArrayBytes(indices,sizeof(std::uint32_t),mesh)
      || !addArrayBytes(elements,sizeof(FieldElement),field)
      || !addArrayBytes(attributeCount,sizeof(AttributeData)+kMaximumNameBytes+1,attributes)
      // This subset preserves serialized attributes, and cannot synthesize new
      // attribute arrays. Twice all wire bytes bounds their logical capacities.
      || !addArrayBytes(binary,2,attributes)
      || !checkedAdd(mesh,attributes,replay) || !checkedAdd(replay,field,replay)
      || !checkedAdd(replay,sizeof(ValueDescriptor)+sizeof(StableId)*4,replay)
      || !addArrayBytes(operations,sizeof(OperationRecord)+kMaximumNameBytes+1,descriptor)
      || !addArrayBytes(attributeCount,sizeof(AttributeDescriptor)+kMaximumNameBytes+1,descriptor)
      || !checkedMul(binary,16,sources)
      || !checkedAdd(descriptor,mesh,descriptorBytes)
      || !checkedAdd(descriptorBytes,attributes,descriptorBytes)
      || !checkedAdd(mesh,sizeof(OperationRecord),waterBytes)) goto overflow;
  out={};
  out.entryCountChecks=13;
  if (!checkedAdd(bindings.size(),Scene::kMaxLights,out.entryRecordVisits)
      || !deriveSceneWork(bindings.size(),scene.textureTexels.size(),environmentBytes,
          text,childRecordBytes,out)) goto overflow;
  out.retainedBytes=sceneBytes;
  if (!checkedAdd(out.retainedBytes,bindings.size(),out.retainedBytes)) goto overflow;
  // Two descriptors cover decoded and cloned immutable admission. Sources and
  // strings use a deliberately conservative sixteen-byte expansion per wire byte.
  if (!addArrayBytes(bindings.size(),childBytes,out.retainedBytes)
      || !checkedAdd(out.retainedBytes,text,out.retainedBytes)
      || !addArrayBytes(descriptorCopies,descriptorBytes,out.retainedBytes)
      || !checkedAdd(out.retainedBytes,sources,out.retainedBytes)) goto overflow;
  out.attemptBytes=out.retainedBytes;
  // Parser/lowered staging, descriptor clones, all retained replay maps, water
  // rest meshes/heights/operation copies and three complete candidate scenes.
  if (!addArrayBytes(128,3*sizeof(std::size_t),out.attemptBytes)
      || !addArrayBytes(3,sceneBytes,out.attemptBytes)
      || !addArrayBytes(4,binary,out.attemptBytes)
      || !addArrayBytes(replayCopies,replay,out.attemptBytes)
      || !addArrayBytes(operations,waterBytes,out.attemptBytes)
      || !addArrayBytes(waterElements,sizeof(double)*2+sizeof(StableId),out.attemptBytes)
      || !addArrayBytes(dependencyElements,8*sizeof(StableId),out.attemptBytes)
      || !addArrayBytes(operations,16*sizeof(std::map<StableId,StableId>),out.attemptBytes)
      || !checkedMul(text,8,out.decodeByteVisits)
      || !checkedMul(operations,3,out.replayIterations)
      || !checkedMul(operations,3,out.admissionReplayIterations)
      || !checkedMul(dependencyElements,9,out.planPairVisits)
      || !checkedMul(operations,elements,out.fieldElementVisits)
      || !checkedMul(out.fieldElementVisits,8,out.fieldElementVisits)
      || !checkedMul(operations,indices,out.normalIndexVisits)
      || !checkedMul(out.normalIndexVisits,8,out.normalIndexVisits)
      || !checkedMul(operations,vertices,out.waterSampleElements)) goto overflow;
  if (out.attemptBytes>limits.maxBufferBytes) {
    error="Timeline scene family exceeds its logical retained/attempt storage budget"; return false;
  }
  error.clear(); return true;
overflow:
  error="Timeline scene family storage or work envelope overflow"; return false;
}
} // namespace videohelper::geometry::timelinefamily
