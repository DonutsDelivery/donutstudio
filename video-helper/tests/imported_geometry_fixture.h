#pragma once

#include "../src/glb_geometry_core_adapter.h"
#include "../src/geometry_core_backend.h"

inline videohelper::gltf::GlbStaticMeshDocument importedGeometryFixture() {
  videohelper::gltf::GlbStaticMeshDocument asset;
  asset.selectedScene=0; asset.scenes.resize(2); asset.scenes[0].rootNodes={0}; asset.scenes[1].rootNodes={1};
  asset.nodes.resize(2); asset.nodes[0].mesh=1; asset.nodes[1].mesh=0;
  asset.meshes.resize(2); asset.materials.resize(2);
  videohelper::gltf::GlbPrimitiveRecord p;
  p.positions={-0.7f,-0.6f,0,0.7f,-0.6f,0,0,0.7f,0};
  p.normals={0,0,1,0,0,1,0,0,1}; p.texCoords0={0,0,1,0,0.5f,1};
  p.colors0={1,0.2f,0.1f,1,0.1f,1,0.2f,1,0.2f,0.1f,1,1};
  p.indices={0,1,2}; p.material=1;
  asset.meshes[1].primitives.push_back(p);
  return asset;
}

inline videowire::geometry::PortContract importedGeometryContract() {
  using namespace videowire::geometry;
  PortContract c; c.carrier=CarrierKind::geometry3D; c.maxVertices=4096;
  c.maxIndices=12288; c.maxAttributes=kMaximumAttributes; return c;
}

template <typename ReadPixels>
bool verifyNativeImportedGeometry(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  const auto source=videohelper::gltf::adaptGlbMeshToGeometryCore(importedGeometryFixture(),1,9001,9000,1,error);
  if (!source) return false;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  std::vector<std::uint8_t> original;
  for (int variant=0;variant<6;++variant) {
    auto value=source->descriptor();
    if (variant==1) {
      PortContract field; field.carrier=CarrierKind::field; field.fieldDomain=Domain::vertex;
      field.fieldValueType=ValueType::vector; field.fieldInterpolation=Interpolation::constant;
      field.maxFieldElements=4096; field.maxAttributes=kMaximumAttributes;
      const auto query=lowerElementField(value,9002,OperationCode::positionField,field,error);
      const auto displaced=query ? lowerVectorDisplacement(value,*query,9003,0.5f,
          withAttributeContract(importedGeometryContract(),value),error) : std::nullopt;
      if (!displaced) return false;
      value=*displaced;
    } else if (variant==2) {
      spectrum::Binding binding; binding.deformer.mode=spectrum::Deformation::vector;
      binding.direction={1,0,0}; binding.gain=0.6f;
      const auto deformed=lowerAudioDeformer(value,9004,binding,importedGeometryContract(),error);
      if (!deformed) return false;
      value=*deformed;
    } else if (variant==3 || variant==5) {
      PortContract pointContract; pointContract.carrier=CarrierKind::points3D; pointContract.maxPoints=4096;
      const auto points=lowerPointsFromVertices(value,9006,pointContract,error);
      Transform transform; transform.scale={0.35f,0.35f,0.35f};
      const auto small=lowerTransformGeometry(value,9007,transform,importedGeometryContract(),error);
      PortContract instanceContract; instanceContract.carrier=CarrierKind::instances3D; instanceContract.maxInstances=4096;
      auto instances=points && small ? lowerInstanceOnPoints(*points,small->stableId,9008,instanceContract,error) : std::nullopt;
      if (!instances) return false;
      auto operations=small->operations;
      if (!appendUniqueOperations(operations,instances->operations,error)) return false;
      instances->operations=std::move(operations); instances->dispatchCount=instances->operations.size();
      value=*instances;
    }
    auto contract=withAttributeContract(importedGeometryContract(),value);
    if (variant==3 || variant==5) {
      contract.carrier=CarrierKind::instances3D; contract.maxInstances=4096;
      contract.maxVertices=0; contract.maxIndices=0;
    }
    if (variant>=4) {
      const auto assigned=lowerMaterialTableSlot(value,9010,9011,1,{0.1,0.2,1,0,0,0.3,0.2,0.8},1,contract,error);
      if (!assigned) return false;
      value=*assigned;
    }
    const auto admitted=admitValue(value,contract,error);
    if (!admitted) return false;
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    GeometryCorePlanRuntime runtime(capabilities);
    const auto preview=runtime.admitPreview({1,1,1,9005,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,9005,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    SpectrumEvaluation audio; audio.timeSeconds=1; audio.bands.fill(0.8f);
    audio.featuresAt=[](double) { spectrum::Bands bands; bands.fill(0.8f); return bands; };
    auto exportAudio=audio;
    const auto draw=executeNativeGeometry(backend,*preview,64,64,error,false,{},&audio);
    const auto exportDraw=executeNativeGeometry(backend,*exported,64,64,error,false,{},&exportAudio);
    if (!draw || !exportDraw) return false;
    const auto pixels=readPixels(draw->frame);
    if (pixels.empty() || pixels!=readPixels(exportDraw->frame)
        || (variant!=3 && variant!=5 && draw->stats.ordinaryDrawCount!=1)
        || (variant==5 && draw->stats.ordinaryDrawCount!=3)
        || (variant==3 && (draw->stats.instancedDrawCount!=1 || draw->stats.submittedInstanceCount!=3))
        || (variant!=0 && pixels==original)) {
      error="Imported mesh fields and audio deformation must change native pixels with preview/export equality"; return false;
    }
    if (variant==0) original=pixels;
  }
  return true;
}
