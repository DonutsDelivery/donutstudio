#pragma once

#include "../src/geometry_core_backend.h"

inline videowire::geometry::PortContract distributedInstanceContract() {
  videowire::geometry::PortContract result;
  result.carrier=videowire::geometry::CarrierKind::instances3D;
  result.overflow=videowire::geometry::OverflowPolicy::reject;
  result.maxInstances=64;
  return result;
}

inline std::optional<videowire::geometry::ValueDescriptor>
distributedInstanceFixture(videowire::geometry::OperationCode distribution,
                           unsigned seed, bool edited, std::string &error) {
  using namespace videowire::geometry;
  PortContract meshContract;
  meshContract.carrier=CarrierKind::geometry3D; meshContract.overflow=OverflowPolicy::reject;
  meshContract.maxVertices=24; meshContract.maxIndices=36;
  PortContract pointsContract;
  pointsContract.carrier=CarrierKind::points3D; pointsContract.overflow=OverflowPolicy::reject;
  pointsContract.maxPoints=64;
  const auto cube=lowerMeshGenerator(7101,7101,1,OperationCode::cube,{2,2,1,0},meshContract,error);
  const auto prototype=lowerMeshGenerator(7102,7102,1,OperationCode::cube,{0.22f,0.22f,0.22f,0},meshContract,error);
  auto points=cube ? lowerPointDistribution(*cube,7103,distribution,16,seed,pointsContract,error) : std::nullopt;
  auto instances=points && prototype ? lowerInstanceOnPoints(*points,prototype->stableId,7104,
                                                              distributedInstanceContract(),error) : std::nullopt;
  if (!instances) return std::nullopt;
  instances->operations.insert(instances->operations.begin(),prototype->operations.begin(),prototype->operations.end());
  instances->dispatchCount=instances->operations.size();
  if (!edited) return instances;
  Transform transform;
  transform.translation={0.6f,0.3f,0}; transform.scale={1.6f,0.6f,1};
  transform.rotation={0,0,0.38268343f,0.9238795f};
  auto moved=lowerInstanceEdit(*instances,7105,OperationCode::transformInstances,{0,15,2,0},
                               transform,distributedInstanceContract(),error);
  return moved ? lowerInstanceEdit(*moved,7106,OperationCode::cullInstances,{1,15,3,0},{},
                                    distributedInstanceContract(),error) : std::nullopt;
}

template <typename ReadPixels>
bool verifyNativeDistributedInstances(arbitgpu::NativeFixtureSceneBackend &backend,
                                      ReadPixels readPixels, std::string &error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  for (const auto code : {OperationCode::pointsOnFaces,OperationCode::pointsInBoxVolume}) {
    std::vector<std::uint8_t> original;
    for (unsigned variant=0;variant<3;++variant) {
      auto value=distributedInstanceFixture(code,variant==2 ? 197 : 29,variant==1,error);
      const auto admitted=value ? admitValue(*value,distributedInstanceContract(),error) : std::nullopt;
      if (!admitted) return false;
      const auto bytes=lowerRuntimePlan(distributedInstanceContract(),*admitted);
      GeometryCorePlanRuntime runtime(capabilities);
      const auto preview=runtime.admitPreview({1,1,1,7106,1,PlanUse::preview},bytes,{}, {},error);
      const auto exported=runtime.admitExport({1,1,1,7106,1,PlanUse::exportRender},bytes,{}, {},error);
      if (!preview || !exported) return false;
      const auto draw=executeNativeGeometry(backend,*preview,64,64,error);
      const auto exportDraw=executeNativeGeometry(backend,*exported,64,64,error);
      if (!draw || !exportDraw) return false;
      const auto pixels=readPixels(draw->frame);
      std::size_t visible=0;
      for (std::size_t i=0;i<pixels.size();i+=4)
        if (std::abs(int(pixels[i])-int(pixels[0]))+std::abs(int(pixels[i+1])-int(pixels[1]))
            +std::abs(int(pixels[i+2])-int(pixels[2]))>5) ++visible;
      if (pixels.empty() || visible<10 || pixels!=readPixels(exportDraw->frame)
          || draw->drawnInstanceIds!=exportDraw->drawnInstanceIds
          || draw->drawnInstanceIds.size()!=(variant==1 ? 11u : 16u)
          || draw->stats.instancedDrawCount!=1 || draw->stats.submittedInstanceCount!=draw->drawnInstanceIds.size()
          || draw->stats.vertexBytes!=24*sizeof(HarmonicMIDI::grid::SceneVertex)
          || draw->stats.indexBytes!=36*sizeof(std::uint32_t)
          || (variant!=0 && pixels==original)) {
        error="Distributed instances must produce visible seed and transform changes with one source mesh and equal preview/export pixels";
        return false;
      }
      if (variant==0) original=pixels;
    }
  }
  error.clear(); return true;
}
