#pragma once

#include "../../shared/GeometryCoreScene.h"
#include "../src/gpu_backend/backend.h"

inline std::shared_ptr<HarmonicMIDI::grid::Visual3DScene>
geometryCompositionFixture(std::string& error) {
  using namespace videowire::geometry;
  using namespace HarmonicMIDI::grid;
  PortContract meshContract;
  meshContract.carrier = CarrierKind::geometry3D;
  meshContract.overflow = OverflowPolicy::reject;
  meshContract.maxVertices = 4096;
  meshContract.maxIndices = 12288;
  auto cube = lowerMeshGenerator(501,501,1,OperationCode::cube,{0.8f,0.8f,0.8f,0},meshContract,error);
  auto sphere = lowerMeshGenerator(502,502,1,OperationCode::icosphere,{0.16f,1,0,0},meshContract,error);
  auto grid = lowerGrid(503,503,1,2,2,0.5f,meshContract,error);
  if (!cube || !sphere || !grid) return {};
  PortContract pointContract;
  pointContract.carrier = CarrierKind::points3D;
  pointContract.overflow = OverflowPolicy::reject;
  pointContract.maxPoints = 64;
  PortContract instanceContract;
  instanceContract.carrier = CarrierKind::instances3D;
  instanceContract.overflow = OverflowPolicy::reject;
  instanceContract.maxInstances = 64;
  auto points = lowerPointsFromVertices(*grid,504,pointContract,error);
  if (!points) return {};
  auto instances = lowerInstanceOnPoints(*points,sphere->stableId,505,instanceContract,error);
  if (!instances) return {};
  instances->operations.insert(instances->operations.begin(),sphere->operations.begin(),sphere->operations.end());
  instances->dispatchCount += sphere->dispatchCount;
  auto cubeValue=admitValue(*cube,meshContract,error);
  auto instanceValue=admitValue(*instances,instanceContract,error);
  if (!cubeValue || !instanceValue) return {};
  auto scene=std::make_shared<Visual3DScene>();
  scene->id={7}; scene->activeCamera={50}; scene->ambientColor={0.2f,0.2f,0.2f};
  scene->cameras[0].id={50}; scene->cameras[0].transform.translation.z=4;
  scene->cameraCount=1;
  scene->lights[0].id={40}; scene->lights[0].intensity=1.5f; scene->lightCount=1;
  SceneObjectRecord object;
  object.id={10}; object.transform.translation.x=-0.75f;
  SceneMaterialRecord material;
  material.id={20}; material.baseColor={1,0.03f,0.02f}; material.emissive={0.1f,0,0};
  if (!appendGeometryToScene(*cubeValue,object,material,*scene,error)) return {};
  object.id={11}; object.transform.translation.x=0.65f;
  material.id={21}; material.baseColor={0.02f,0.03f,1}; material.emissive={0,0,0.1f};
  if (!appendGeometryToScene(*instanceValue,object,material,*scene,error)
      || !validateVisual3DScene(*scene).valid()) return {};
  return scene;
}

template<class ReadPixels>
bool verifyNativeGeometryComposition(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, bool bgra, std::string& error) {
  const auto reject=[&](const char* message) { error=message; return false; };
  auto scene=geometryCompositionFixture(error);
  if (!scene) return false;
  if (scene->vertexCount!=66 || scene->objectCount!=6 || scene->materialCount!=2)
    return reject("composed mesh and instances lost exact shared geometry or object identity");
  auto preparation=backend.prepare(scene);
  if (!preparation.prepared) { error=preparation.error; return false; }
  auto preview=backend.render(scene,preparation.resources,128,128);
  auto exported=backend.render(scene,preparation.resources,128,128);
  if (!preview.rendered || !exported.rendered || preview.stats.ordinaryDrawCount!=5)
    return reject("composed geometry did not draw one cube and four independent sphere instances");
  const auto color=readPixels(preview.frame);
  if (color.size()!=128u*128u*4u || color!=readPixels(exported.frame))
    return reject("composed geometry preview and export differ");
  std::size_t redPixels=0,bluePixels=0;
  for (std::size_t offset=0;offset<color.size();offset+=4) {
    const auto red=color[offset+(bgra?2:0)],blue=color[offset+(bgra?0:2)];
    if (red>blue+20) ++redPixels;
    if (blue>red+20) ++bluePixels;
  }
  if (redPixels<30 || bluePixels<30)
    return reject("composed geometry did not retain visible red and blue object materials");
  for (const auto output : {renderpassoutput::Output::Depth,renderpassoutput::Output::Normal,
      renderpassoutput::Output::Motion,renderpassoutput::Output::Emission,renderpassoutput::Output::Mask,
      renderpassoutput::Output::ObjectId,renderpassoutput::Output::MaterialId}) {
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.imageOutput=output;
    auto inspected=backend.render(scene,preparation.resources,128,128,inputs);
    auto repeated=backend.render(scene,preparation.resources,128,128,inputs);
    if (!inspected.rendered || !repeated.rendered)
      return reject("composed geometry inspection did not render");
    const auto pixels=readPixels(inspected.frame);
    if (pixels.size()!=color.size() || pixels==color || pixels!=readPixels(repeated.frame))
      return reject("composed geometry inspection lost output selection or preview/export parity");
  }
  auto moved=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*scene);
  moved->objects[0].transform.translation.y=0.75f;
  auto movedPreparation=backend.prepare(moved);
  auto movedFrame=movedPreparation.prepared
      ? backend.render(moved,movedPreparation.resources,128,128)
      : arbitgpu::NativeFixtureSceneSubmission{};
  if (!movedFrame.rendered || readPixels(movedFrame.frame)==color || readPixels(preview.frame)!=color)
    return reject("independent object edits failed to move geometry or overwrote the retained frame");
  error.clear();
  return true;
}
