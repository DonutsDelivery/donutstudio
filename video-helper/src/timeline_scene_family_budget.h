#pragma once

#include "../../shared/GeometryTimelineSceneBinding.h"

#include <cstring>
#include <limits>
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
      || !sameVector(a.ambientColor,b.ambientColor) || !sameSceneVolume(a.worldVolume,b.worldVolume)
      || a.vertexCount!=b.vertexCount || a.indexCount!=b.indexCount
      || a.objectCount!=b.objectCount || a.materialCount!=b.materialCount
      || a.textureCount!=b.textureCount || a.textureTexelCount!=b.textureTexelCount
      || a.lightCount!=b.lightCount || a.cameraCount!=b.cameraCount
      || a.textureTexels.size()!=b.textureTexels.size()
      || a.vertices.size()!=b.vertices.size() || a.objects.size()!=b.objects.size()
      || a.materials.size()!=b.materials.size() || a.textures.size()!=b.textures.size()
      || a.lights.size()!=b.lights.size() || a.cameras.size()!=b.cameras.size()) return false;
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
        || !HarmonicMIDI::grid::sameSceneVolume(x.volume,y.volume)
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
        || x.environmentCameraVisible!=y.environmentCameraVisible
        || !sameCameraBackground(x.cameraBackground,y.cameraBackground)
        || !sameEnvironmentImage(x.environmentImage,y.environmentImage)
        || !sameSpectralSource(x.spectralSource,y.spectralSource)
        || !sameLightShapeAndVisibility(x,y)) return false;
  }
  for (std::size_t i=0;i<a.cameras.size();++i) {
    const auto& x=a.cameras[i]; const auto& y=b.cameras[i];
    if (x.id!=y.id || !sameTransform(x.transform,y.transform)
        || !sameFloat(x.verticalFovRadians,y.verticalFovRadians)
        || !sameFloat(x.nearPlane,y.nearPlane) || !sameFloat(x.farPlane,y.farPlane)
        || !sameFloat(x.orthographicHeight,y.orthographicHeight)
        || !sameCameraLens(x,y)) return false;
  }
  return true;
}

// Source schedule for validateVisual3DScene, sameScene and the binding joins.
// Use allocated lengths even for sparse scenes: replay/default construction can
// visit records absent from the authored counts. Early returns only reduce work.
// These are visit estimates, not sizes, so they saturate: a huge scene's cubic
// worst case is an estimate that stays at the top of size_t, never a failure.
inline bool visitAdd(std::size_t a,std::size_t b,std::size_t& out) noexcept {
  out=a>std::numeric_limits<std::size_t>::max()-b ? std::numeric_limits<std::size_t>::max() : a+b;
  return true;
}
inline bool visitMul(std::size_t a,std::size_t b,std::size_t& out) noexcept {
  out=b!=0 && a>std::numeric_limits<std::size_t>::max()/b ? std::numeric_limits<std::size_t>::max() : a*b;
  return true;
}
inline bool visitAddMul(std::size_t a,std::size_t b,std::size_t& out) noexcept {
  std::size_t product=0; visitMul(a,b,product); return visitAdd(out,product,out);
}
inline bool deriveSceneWork(const Scene& scene,std::size_t children,std::size_t texels,
    std::size_t environmentBytes,std::size_t text,std::size_t childRecordBytes,
    Envelope& out) noexcept {
  const auto O=scene.objects.size(), M=scene.materials.size(), T=scene.textures.size();
  const auto L=scene.lights.size(), C=scene.cameras.size();
  const auto V=scene.vertices.size(), I=scene.indices.size();
  constexpr auto S=diffractionmaterial::kMaximumSpectralSamples;
  std::size_t calls=0,perValidation=0,searches=0,depths=0,multiplies=0;
  std::size_t completeRecords=0,bindingVisits=0,copies=0,stringPasses=0,terms=0;
  // Initial admission, one validTimelineSceneBinding per admitted child, then
  // two validations per future replacement. Cache reuse needs only the first.
  if (!visitMul(children,3,calls) || !visitAdd(calls,1,calls)
      || !visitMul(O,O+1,depths) || !visitMul(O,O,multiplies)
      || !visitAdd(depths,multiplies,terms)
      || !visitMul(terms,O,searches) || !visitMul(calls,searches,out.hierarchyLookupVisits)
      || !visitMul(calls,terms,out.hierarchyDepthIterations)
      || !visitAdd(multiplies,O,terms) || !visitMul(calls,terms,out.matrixConstructions)
      || !visitMul(calls,multiplies,terms) || !visitMul(terms,4*4*4,out.matrixMultiplyTerms)
      // Each matrixFor initializes and transfers 16 elements. Each multiply
      // zeroes 16 and assigns 16 world elements. Postchecks read 9 scale, 18 dot and 3
      // translation elements. Each of the 64 terms reads two and updates one
      // result element. Arithmetic inside each term is fixed.
      || !visitMul(out.matrixConstructions,16+16,out.matrixElementVisits)
      || !visitAddMul(terms,16+16+3*4*4*4,out.matrixElementVisits)
      || !visitMul(O,9+18+3,terms)
      || !visitAddMul(calls,terms,out.matrixElementVisits)
      || !visitMul(O,3+3,terms)
      || !visitMul(calls,terms,out.matrixPostcheckIterations)) return false;
  out.sceneValidationCalls=calls;
  // All duplicateIdAt earlier-record scans, parent-presence lookup, both
  // hierarchy searches, object material lookup, five material texture lookups
  // and the active-camera lookup. Object index ranges can overlap, so validation
  // must reserve O*I index checks rather than assume one pass over indices.
  for (const auto count : {O,M,T,L,C})
    if (count!=0) visitAddMul(count,count-1,perValidation);
  perValidation/=2;
  // Each term below is at most a few times a live length, which cannot wrap:
  // every length is held in memory by records larger than this multiplier.
  if (!visitAdd(perValidation,searches,perValidation)
      || !visitAddMul(O,O+M+I,perValidation)
      || !visitAddMul(M,5*T,perValidation)
      || !visitAdd(perValidation,C,perValidation)
      || !visitAdd(perValidation,1+T+M+2*O+V+2*L+C,perValidation)
      // Three sky channel checks; two spectral arrays and directional
      // coefficients have S entries each, including version-zero comparisons.
      || !visitAdd(calls,1,terms)
      || !visitMul(terms,L*(3+3*S),out.lightChannelVisits)
      || !visitMul(calls,perValidation,out.sceneRecordVisits)
      || !visitAdd(V+I+O+M+T+L+C,texels,completeRecords)
      // One exact full-scene cache comparison. Three binding range scans:
      // initial admission, admitted baseline and future replacement baseline.
      || !visitMul(children,3*O,out.rangeComparisons)
      // Five object searches: initial/range, baseline, retained child, future
      // baseline and future replacement. Three material searches accompany the
      // admission baseline and both future baseline/replacement joins. Three
      // binding-record passes cover canonical order, exact plan equality and
      // backend equality. Each fixed scene header is one record visit.
      || !visitMul(children,5*O+3*M+2*V+3*I+3,bindingVisits)
      || !visitAdd(out.sceneRecordVisits,completeRecords+1,out.sceneRecordVisits)
      || !visitAdd(out.sceneRecordVisits,out.rangeComparisons,out.sceneRecordVisits)
      || !visitAdd(out.sceneRecordVisits,bindingVisits,out.sceneRecordVisits)
      // Frozen copy, future composed-scene copy, and three default candidate
      // scenes per child. Include every fixed record and texture element.
      || !visitMul(children,3,copies) || !visitAdd(copies,2,copies)
      || !visitMul(copies,completeRecords,out.sceneCopyElements)
      // Admission/future baseline append and future evaluated append each
      // write vertices and indices, then replacement writes vertices once.
      || !visitAddMul(children,4*V+3*I,out.sceneCopyElements)
      // Validator strings, full cache equality, scene copies, binding equality
      // and retention. Backend identity comparisons are bounded by child bytes.
      || !visitAdd(calls,copies,stringPasses) || !visitAdd(stringPasses,1,stringPasses)
      || !visitMul(stringPasses,environmentBytes,out.sceneStringByteVisits)
      || !visitAddMul(text,2,out.sceneStringByteVisits)
      || !visitAddMul(children,childRecordBytes,out.sceneStringByteVisits)) return false;
  return true;
}

// Reserve the worst admitted mesh/field per operation before parsing. Serialized
// payload size bounds source arrays and strings; retained sources cannot exceed
// that payload. Replay maps may retain a mesh, attributes and a field per op.
inline bool deriveEnvelope(const Scene& scene,const std::vector<TimelineSceneBinding>& bindings,
    const ResourceLimits& limits,std::size_t childRecordBytes,Envelope& out,std::string& error) {
  std::size_t text=0,binary=0,operations=0,environmentBytes=0;
  // Entry schedule: 13 count predicates, every binding's size and identity
  // record, and every light's string capacity. Contents are untouched.
  if (bindings.empty()
      || scene.vertexCount>scene.vertices.size() || scene.indexCount>scene.indices.size()
      || scene.objectCount>scene.objects.size() || scene.materialCount>scene.materials.size()
      || scene.textureCount>scene.textures.size() || scene.lightCount>scene.lights.size()
      || scene.cameraCount>scene.cameras.size()
      || scene.textureTexelCount!=scene.textureTexels.size()
      || scene.vertexCount>limits.maxVertices || scene.indexCount>limits.maxIndices) {
    error="Timeline scene family needs at least one binding and scene counts that match its allocated records"; return false;
  }
  for (const auto& binding : bindings) {
    const auto n=binding.loweredPlan.size();
    if (!binding.object.isValid() || n==0 || n%2!=0
        || !checkedAdd(text,n,text) || !checkedAdd(binary,n/2,binary)) {
      error="Timeline scene family has an invalid lowered plan or its plan sizes overflow"; return false;
    }
    // Every operation has at least 41 wire bytes. This upper bound does not
    // inspect untrusted operation records or allocate decoder staging.
    if (!checkedAdd(operations,n/2/41,operations)) return false;
  }
  operations=std::max(operations,bindings.size());
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
  // Live scene lengths. A generator that outgrows them allocates what it needs
  // when it runs, and a failed allocation reports itself there.
  const auto vertices=scene.vertices.size();
  const auto indices=scene.indices.size();
  const auto elements=scene.vertices.size();
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
  if (!checkedAdd(bindings.size(),scene.lights.size(),out.entryRecordVisits)
      || !deriveSceneWork(scene,bindings.size(),scene.textureTexels.size(),environmentBytes,
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
  if (!addArrayBytes(operations,3*sizeof(std::size_t),out.attemptBytes)
      || !addArrayBytes(3,sceneBytes,out.attemptBytes)
      || !addArrayBytes(4,binary,out.attemptBytes)
      || !addArrayBytes(replayCopies,replay,out.attemptBytes)
      || !addArrayBytes(operations,waterBytes,out.attemptBytes)
      || !addArrayBytes(waterElements,sizeof(double)*2+sizeof(StableId),out.attemptBytes)
      || !addArrayBytes(dependencyElements,8*sizeof(StableId),out.attemptBytes)
      || !addArrayBytes(operations,16*sizeof(std::map<StableId,StableId>),out.attemptBytes)
      || !visitMul(text,8,out.decodeByteVisits)
      || !visitMul(operations,3,out.replayIterations)
      || !visitMul(operations,3,out.admissionReplayIterations)
      || !visitMul(dependencyElements,9,out.planPairVisits)
      || !visitMul(operations,elements,out.fieldElementVisits)
      || !visitMul(out.fieldElementVisits,8,out.fieldElementVisits)
      || !visitMul(operations,indices,out.normalIndexVisits)
      || !visitMul(out.normalIndexVisits,8,out.normalIndexVisits)
      || !visitMul(operations,vertices,out.waterSampleElements)) goto overflow;
  if (out.attemptBytes>limits.maxBufferBytes) {
    error="Timeline scene family needs "+std::to_string(out.attemptBytes)+" bytes and its caller allows "
        +std::to_string(limits.maxBufferBytes); return false;
  }
  error.clear(); return true;
overflow:
  error="Timeline scene family storage size overflows size_t: "+std::to_string(bindings.size())+" bindings, "
      +std::to_string(operations)+" operations, "+std::to_string(scene.vertices.size())+" vertices, "
      +std::to_string(scene.indices.size())+" indices"; return false;
}
} // namespace videohelper::geometry::timelinefamily
