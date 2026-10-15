#pragma once

#include "../src/composed_timeline_geometry.h"
#include "../src/geometry_core_backend.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

struct DirectionalWaveFixtureSettings final {
  double amplitude=0.2,wavelength=2,directionDegrees=30,speed=0.5,phaseRadians=0.3,loopPeriod=4;
};

struct WaterSurfaceFixtureSettings final {
  double speed=0,damping=60,scoreGain=2,audioGain=0;
};

inline std::optional<visualimportedscenerender::Request> composedTimelineGeometryFixture(
    std::string& error, const DirectionalWaveFixtureSettings* wave=nullptr,
    const WaterSurfaceFixtureSettings* water=nullptr) {
  using namespace videowire::geometry;
  using namespace HarmonicMIDI::grid;
  PortContract geometry; geometry.carrier = CarrierKind::geometry3D;
  geometry.maxVertices = 4096; geometry.maxIndices = 12288;
  PortContract vector; vector.carrier = CarrierKind::field; vector.fieldDomain = Domain::vertex;
  vector.fieldValueType = ValueType::vector; vector.fieldInterpolation = Interpolation::constant; vector.maxFieldElements = 4096;
  auto scalar = vector; scalar.fieldValueType = ValueType::floatValue;
  auto grid = lowerGrid(601,601,1,wave ? 5 : 2,wave ? 5 : 2,wave ? 0.35f : 1,geometry,error);
  std::optional<ValueDescriptor> mesh;
  if (water && grid)
    mesh=lowerWaterSurfaceSimulation(*grid,606,608,water->speed,water->damping,
        water->scoreGain,water->audioGain,geometry,error);
  else if (wave && grid)
    mesh=lowerDirectionalSurfaceWave(*grid,606,10000,wave->amplitude,wave->wavelength,wave->directionDegrees,
        wave->speed,wave->phaseRadians,wave->loopPeriod,geometry,error);
  else {
    auto positions = grid ? lowerElementField(*grid,602,OperationCode::positionField,vector,error) : std::nullopt;
    FieldOperationSettings settings; settings.domain = Domain::vertex; settings.valueType = ValueType::floatValue; settings.mode = 3;
    auto x = positions ? lowerConstructedField(603,OperationCode::fieldVectorMeasure,settings,&*positions,nullptr,nullptr,scalar,error) : std::nullopt;
    settings.mode = 2; settings.values = {4,0,0.25,0,0.25,0,0,0};
    auto time = x ? lowerConstructedField(604,OperationCode::fieldTimeline,settings,&*x,nullptr,nullptr,scalar,error) : std::nullopt;
    settings.valueType = ValueType::vector; settings.mode = 1; settings.values = {0,0,1,0,0,0,0,0};
    auto offset = time ? lowerConstructedField(605,OperationCode::fieldComposeVector,settings,&*time,nullptr,nullptr,vector,error) : std::nullopt;
    mesh = offset ? lowerVectorDisplacement(*grid,*offset,606,1,geometry,error) : std::nullopt;
  }
  auto value = mesh ? admitValue(*mesh,geometry,error) : std::nullopt;
  if (!value) return std::nullopt;
  auto scene = std::make_shared<Visual3DScene>();
  scene->id = {7}; scene->activeCamera = {50}; scene->ambientColor = {0.01f,0.02f,0.03f};
  scene->cameras[0].id = {50}; scene->cameras[0].transform.translation.z = 4; scene->cameraCount = 1;
  scene->lights[0].id = {40}; scene->lights[0].kind = SceneLightKind::Directional;
  scene->lights[0].transform.rotation = {-0.25f,0.25f,0,0.9354143f}; scene->lights[0].intensity = 1.5f;
  scene->lights[1].id = {41}; scene->lights[1].kind = SceneLightKind::Environment;
  scene->lights[1].color = {0.2f,0.3f,0.4f}; scene->lights[1].intensity = 0.3f; scene->lightCount = 2;
  SceneObjectRecord object; object.id = {10}; object.transform.translation.x = -0.65f;
  SceneMaterialRecord material; material.id = {20}; material.baseColor = {1,0.03f,0.02f}; material.roughness = 0.8f;
  if (!appendGeometryToScene(*value,object,material,*scene,error)) return std::nullopt;
  auto cube = lowerMeshGenerator(607,607,1,OperationCode::cube,{0.7f,0.7f,0.7f,0},geometry,error);
  auto cubeValue = cube ? admitValue(*cube,geometry,error) : std::nullopt;
  object.id = {11}; object.transform.translation.x = 0.7f;
  material.id = {21}; material.baseColor = {0.02f,0.03f,1};
  if (!cubeValue || !appendGeometryToScene(*cubeValue,object,material,*scene,error)
      || !validateVisual3DScene(*scene).valid()) return std::nullopt;
  visualimportedscenerender::Request request;
  request.sourceStableId = 7; request.renderStableId = 12;
  request.structuralRevision = 9; request.evaluationRevision = 12; request.sceneSnapshot = scene;
  request.timelineGeometry = TimelineSceneBinding{{10},encodeLoweredPlanText(lowerRuntimePlan(geometry,*value))};
  return request;
}

inline bool verifyComposedTimelinePositions(const HarmonicMIDI::grid::Visual3DScene& frame,
    const HarmonicMIDI::grid::Visual3DScene& authored, std::string& error) {
  if (frame.vertexCount != 28 || frame.indexCount != 42 || frame.objectCount != 2
      || frame.materialCount != 2 || frame.lightCount != 2 || frame.cameraCount != 1) {
    error = "Timeline composition changed authored scene cardinality"; return false;
  }
  const std::array<std::uint32_t,6> indices {0,1,2,1,3,2};
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& p = frame.vertices[i].position;
    const auto& base = authored.vertices[i].position;
    if (p.x != base.x || p.y != base.y || std::abs(p.z - (p.x < 0 ? 0.0f : 0.25f)) > 0.000001f) {
      error = "Timeline grid must independently evaluate negative X to Z zero and positive X to Z 0.25 at half a second"; return false;
    }
  }
  for (std::size_t i = 0; i < indices.size(); ++i)
    if (frame.indices[i] != indices[i]) { error = "Timeline grid changed its six fixed indices"; return false; }
  for (std::size_t i = 4; i < frame.vertexCount; ++i)
    if (!videowire::geometry::sameSceneVertex(frame.vertices[i],authored.vertices[i])) {
      error = "Timeline evaluation overwrote static cube vertices"; return false;
    }
  auto expected = authored;
  for (std::size_t i = 0; i < 4; ++i) expected.vertices[i] = frame.vertices[i];
  if (sceneaov::detail::encodeScene(expected) != sceneaov::detail::encodeScene(frame)) {
    error = "Timeline evaluation changed authored materials, lights, environment, camera, object transforms or topology"; return false;
  }
  return true;
}

template <typename ReadPixels>
bool verifyNativeComposedTimelineGeometry(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, bool bgra, std::string& error) {
  using namespace videohelper::geometry;
  const auto request = composedTimelineGeometryFixture(error);
  if (!request || !visualimportedscenerender::valid(*request)) return false;
  NativeGeometryCoreCapabilitySource capabilities(backend); GeometryCorePlanRuntime runtime(capabilities);
  const PlanOwnerIdentity previewOwner {3,4,5,8,9,PlanUse::preview};
  auto exportOwner = previewOwner; exportOwner.use = PlanUse::exportRender;
  const auto evaluate = [&](const auto& owner, double seconds) {
    return evaluateComposedTimelineGeometry(runtime,owner,*request->timelineGeometry,request->sceneSnapshot,seconds,error);
  };
  auto before = evaluate(previewOwner,0), changed = evaluate(previewOwner,0.5), replay = evaluate(previewOwner,4.5);
  auto exported = evaluate(exportOwner,0.5);
  if (!before || !changed || !replay || !exported || !verifyComposedTimelinePositions(*changed,*request->sceneSnapshot,error)) return false;
  if (sceneaov::detail::encodeScene(*changed) != sceneaov::detail::encodeScene(*replay)
      || sceneaov::detail::encodeScene(*changed) != sceneaov::detail::encodeScene(*exported)) {
    error = "Seek/loop replay and export lost exact timeline mesh parity"; return false;
  }
  const auto render = [&](const auto& scene) {
    auto prepared = backend.prepare(scene);
    return prepared.prepared ? backend.render(scene,prepared.resources,128,128) : arbitgpu::NativeFixtureSceneSubmission{};
  };
  auto a = render(before), b = render(changed), c = render(exported);
  if (!a.rendered || !b.rendered || !c.rendered || b.stats.ordinaryDrawCount != 2) {
    error = "Timeline composition did not render its deformed grid and static cube"; return false;
  }
  const auto retainedPixels = readPixels(a.frame), changedPixels = readPixels(b.frame);
  if (retainedPixels == changedPixels || changedPixels != readPixels(c.frame)
      || retainedPixels != readPixels(a.frame) || changedPixels.size() != 128u*128u*4u) {
    error = "Timeline composition lacks native frame causality, retained ownership or preview/export parity"; return false;
  }
  std::size_t red = 0, blue = 0;
  for (std::size_t i = 0; i < changedPixels.size(); i += 4) {
    const auto r = changedPixels[i + (bgra ? 2 : 0)], b = changedPixels[i + (bgra ? 0 : 2)];
    red += r > b + 20; blue += b > r + 20;
  }
  if (red < 30 || blue < 30) { error = "Timeline composition did not retain visible authored grid and cube materials"; return false; }
  return true;
}

// Independently evaluate binary64 heights and area-weighted triangle normals.
// The published normals describe the triangulated mesh, not an analytic surface.
inline bool verifyDirectionalWaveGeometry(const HarmonicMIDI::grid::Visual3DScene& frame,
    const HarmonicMIDI::grid::Visual3DScene& authored, const DirectionalWaveFixtureSettings& wave,
    double seconds, std::string& error) {
  constexpr double tau=6.283185307179586476925286766559;
  if (frame.vertexCount!=49 || frame.indexCount!=132 || frame.objectCount!=2) {
    error="Directional wave changed scene cardinality"; return false;
  }
  using D3=std::array<double,3>;
  std::array<D3,25> points{},normals{};
  const auto angle=wave.directionDegrees*tau/360;
  for (std::size_t i=0;i<25;++i) {
    const auto& p=frame.vertices[i].position;
    const auto& base=authored.vertices[i].position;
    const auto h=wave.amplitude*std::sin(tau*((std::cos(angle)*double(base.x)+std::sin(angle)*double(base.y))
        /wave.wavelength-seconds*wave.speed/wave.wavelength)+wave.phaseRadians);
    if (p.x!=base.x || p.y!=base.y || std::abs(double(p.z)-h)>0.000001) {
      error="Directional wave position differs from independent binary64 sine"; return false;
    }
    points[i]={double(base.x),double(base.y),double(static_cast<float>(h))};
  }
  std::size_t index=0;
  for (std::uint32_t row=0;row<4;++row) for (std::uint32_t column=0;column<4;++column) {
    const auto a=row*5+column,b=a+1,c=a+5,d=c+1;
    for (const auto triangle : {std::array<std::uint32_t,3>{a,b,c},std::array<std::uint32_t,3>{b,d,c}}) {
      for (const auto vertex : triangle)
        if (frame.indices[index++]!=vertex) { error="Directional wave changed grid topology"; return false; }
      const auto& p=points[triangle[0]]; const auto& q=points[triangle[1]]; const auto& r=points[triangle[2]];
      const D3 u{q[0]-p[0],q[1]-p[1],q[2]-p[2]},v{r[0]-p[0],r[1]-p[1],r[2]-p[2]};
      const D3 n{u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
      for (const auto vertex : triangle) for (std::size_t axis=0;axis<3;++axis) normals[vertex][axis]+=n[axis];
    }
  }
  for (std::size_t i=0;i<25;++i) {
    const auto& n=normals[i]; const auto length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    const auto& actual=frame.vertices[i].normal;
    if (std::abs(actual.x-n[0]/length)>0.000001 || std::abs(actual.y-n[1]/length)>0.000001
        || std::abs(actual.z-n[2]/length)>0.000001) {
      error="Directional wave normal differs from independent triangle accumulation"; return false;
    }
  }
  for (std::size_t i=25;i<frame.vertexCount;++i)
    if (!videowire::geometry::sameSceneVertex(frame.vertices[i],authored.vertices[i])) {
      error="Directional wave changed the unrelated cube"; return false;
    }
  auto expected=authored;
  for (std::size_t i=0;i<25;++i) expected.vertices[i]=frame.vertices[i];
  if (sceneaov::detail::encodeScene(expected)!=sceneaov::detail::encodeScene(frame)) {
    error="Directional wave changed authored scene settings or indices"; return false;
  }
  return true;
}

template <typename ReadPixels>
bool verifyNativeDirectionalSurfaceWave(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, bool bgra, std::string& error) {
  using namespace videohelper::geometry;
  DirectionalWaveFixtureSettings wave;
  const auto request=composedTimelineGeometryFixture(error,&wave);
  if (!request || !visualimportedscenerender::valid(*request)) return false;
  NativeGeometryCoreCapabilitySource capabilities(backend); GeometryCorePlanRuntime runtime(capabilities);
  const PlanOwnerIdentity previewOwner{3,4,5,18,9,PlanUse::preview};
  auto exportOwner=previewOwner; exportOwner.use=PlanUse::exportRender;
  const auto evaluate=[&](const auto& owner,double seconds) {
    auto scene=evaluateComposedTimelineGeometry(runtime,owner,*request->timelineGeometry,request->sceneSnapshot,seconds,error);
    if (scene && !verifyDirectionalWaveGeometry(*scene,*request->sceneSnapshot,wave,seconds,error)) scene.reset();
    return scene;
  };
  auto zero=evaluate(previewOwner,0),changed=evaluate(previewOwner,0.5),loop=evaluate(previewOwner,4.5);
  auto backward=evaluate(previewOwner,0.25),reset=evaluate(previewOwner,0),exported=evaluate(exportOwner,0.5);
  auto backReplay=evaluate(previewOwner,0.5),negative=evaluate(previewOwner,-3.5);
  if (!zero || !changed || !loop || !backward || !reset || !exported || !backReplay || !negative) return false;
  const auto encoded=sceneaov::detail::encodeScene(*changed);
  for (const auto& same : {loop,exported,backReplay,negative})
    if (sceneaov::detail::encodeScene(*same)!=encoded) { error="Directional wave lost exact seek, loop or export parity"; return false; }
  if (sceneaov::detail::encodeScene(*zero)!=sceneaov::detail::encodeScene(*reset)) {
    error="Directional wave reset differs from authored time zero"; return false;
  }
  const auto render=[&](const auto& scene) {
    auto prepared=backend.prepare(scene);
    return prepared.prepared ? backend.render(scene,prepared.resources,128,128) : arbitgpu::NativeFixtureSceneSubmission{};
  };
  const auto a=render(zero),b=render(changed),c=render(exported);
  if (!a.rendered || !b.rendered || !c.rendered || b.stats.ordinaryDrawCount!=2) {
    error="Directional wave did not use the normal native composed geometry route"; return false;
  }
  const auto before=readPixels(a.frame),pixels=readPixels(b.frame);
  if (before==pixels || pixels!=readPixels(c.frame) || pixels.size()!=128u*128u*4u) {
    error="Directional wave lacks native time causality or preview/export parity"; return false;
  }
  std::size_t red=0,blue=0;
  for (std::size_t i=0;i<pixels.size();i+=4) {
    const auto r=pixels[i+(bgra ? 2 : 0)],b=pixels[i+(bgra ? 0 : 2)];
    red+=r>b+20; blue+=b>r+20;
  }
  if (red<30 || blue<30) { error="Directional wave and unrelated cube materials are not visible"; return false; }
  const auto saved=sceneaov::detail::encodeScene(*request->sceneSnapshot);
  for (std::size_t control=0;control<6;++control) {
    auto settings=wave;
    if (control==0) settings.amplitude=0;
    if (control==1) settings.wavelength=1;
    if (control==2) settings.directionDegrees=90;
    if (control==3) settings.speed=-0.5;
    if (control==4) settings.phaseRadians=1.3;
    if (control==5) settings.loopPeriod=8;
    auto variant=composedTimelineGeometryFixture(error,&settings);
    auto scene=variant ? evaluateComposedTimelineGeometry(runtime,previewOwner,*variant->timelineGeometry,
        variant->sceneSnapshot,0.5,error) : nullptr;
    if (!scene || !verifyDirectionalWaveGeometry(*scene,*variant->sceneSnapshot,settings,0.5,error)) return false;
    const auto rendered=render(scene);
    if (!rendered.rendered || (control<5 && readPixels(rendered.frame)==pixels)
        || (control==5 && sceneaov::detail::encodeScene(*scene)!=encoded)) {
      error="Directional wave control lacks expected native geometry causality"; return false;
    }
  }
  if (sceneaov::detail::encodeScene(*request->sceneSnapshot)!=saved) {
    error="Directional wave evaluation mutated the authored snapshot"; return false;
  }
  return true;
}
