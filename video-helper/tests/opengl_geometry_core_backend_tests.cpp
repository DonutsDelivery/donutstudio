#include "../src/geometry_core_backend.h"
#include "../src/geometry_core_execution_cache.h"
#include "../src/geometry_core_diagnostic_colors.h"
#include "../src/canonical_block_c_frame.h"
#include "../src/harmonic_link_geometry.h"
#include "../src/gl_loader.h"
#include "note_instance_appearance_fixture.h"
#include "spectrum_instancer_fixture.h"
#include "spectral_history_fixture.h"
#include "audio_deformer_fixture.h"
#include "ordered_audio_geometry_fixture.h"
#include "score_field_fixture.h"
#include "timeline_curve_fixture.h"
#include "point_distribution_fixture.h"
#include "geometry_scene_composition_fixture.h"
#include "constructed_field_fixture.h"
#include "imported_geometry_fixture.h"
#include "instance_appearance_fixture.h"
#include "instance_material_fields_fixture.h"
#include "reactive_surface_fixture.h"
#include "reactive_frame_fixture.h"
#include "material_table_fixture.h"
#include "native_geometry_admission_fixture.h"
#include "solid_body_fixture.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace videowire::geometry;

class CapabilitySource final : public videohelper::geometry::BackendCapabilitySource {
public:
  explicit CapabilitySource(arbitgpu::NativeFixtureSceneBackend &backend)
      : source(backend) {}
  videohelper::geometry::BackendCapabilities geometryCoreCapabilities() const override {
    return source.geometryCoreCapabilities();
  }
private:
  videohelper::geometry::NativeGeometryCoreCapabilitySource source;
};

bool fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

std::shared_ptr<HarmonicMIDI::grid::Visual3DScene> ordinarySharedScene() {
  using namespace HarmonicMIDI::grid;
  auto scene = std::make_shared<Visual3DScene>();
  scene->id = {91};
  scene->vertices[0].position = {-0.25f, -0.25f, 0.0f};
  scene->vertices[1].position = {0.25f, -0.25f, 0.0f};
  scene->vertices[2].position = {0.0f, 0.25f, 0.0f};
  for (std::size_t index = 0; index < 3; ++index) {
    scene->vertices[index].normal = {0.0f, 0.0f, 1.0f};
    scene->vertices[index].color = {1.0f, 1.0f, 1.0f, 1.0f};
    scene->indices[index] = static_cast<std::uint32_t>(index);
  }
  scene->vertexCount = 3;
  scene->indexCount = 3;
  scene->materials[0].id = {1};
  scene->materials[0].baseColor = {0.8f, 0.4f, 0.2f};
  scene->materialCount = 1;
  for (std::size_t index = 0; index < 2; ++index) {
    scene->objects[index].id = {static_cast<std::uint32_t>(index + 1)};
    scene->objects[index].material = {1};
    scene->objects[index].vertexCount = 3;
    scene->objects[index].indexCount = 3;
    scene->objects[index].transform.translation.x = index == 0 ? -0.5f : 0.5f;
  }
  scene->objectCount = 2;
  scene->cameras[0].id = {1};
  scene->cameras[0].transform.translation = {0.0f, 0.0f, 5.0f};
  scene->cameras[0].verticalFovRadians = 0.7853981634f;
  scene->cameras[0].nearPlane = 0.1f;
  scene->cameras[0].farPlane = 1000.0f;
  scene->cameraCount = 1;
  scene->activeCamera = {1};
  return scene;
}

arbitgpu::NativeFixtureSceneRuntimeInputs twoNoteRuntimeInputs() {
  auto score = std::make_shared<arbitmod::Score>();
  score->notationVersion = 1;
  score->scoreRevision = 1;
  score->rootFreq = 440.0f;
  for (int index = 0; index < 2; ++index) {
    arbitmod::Note note;
    note.id = index + 1;
    note.trackId = 1;
    note.startBeat = static_cast<float>(index);
    note.lengthBeats = 4.0f;
    note.velocity = 100.0f;
    note.freqHz = index == 0 ? 440.0f : 660.0f;
    note.durationSeconds = 2.0f;
    score->notes.push_back(note);
  }
  canonicalblockc::FrameKey key;
  key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
  key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
  key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = 1;
  key.seekGeneration = 1;
  key.fps = 60.0;
  canonicalblockc::FrameProducer producer;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.canonicalBlockCFrame = producer.evaluate(key, score, 0.0f);
  visualnoteinstancing::Mapping mapping;
  mapping.x = visualnoteinstancing::MappingAxis::Onset;
  mapping.xScale = 0.2f;
  mapping.meshScale = 0.8f;
  inputs.noteInstanceMapping = mapping;
  return inputs;
}
} // namespace

int main() {
  if (!glfwInit()) {
    std::cerr << "SKIP: GLFW initialization failed\n";
    return 77;
  }
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  auto *window = glfwCreateWindow(64, 64, "geometry-core-pixel", nullptr, nullptr);
  if (!window) {
    std::cerr << "SKIP: OpenGL 3.3 core context creation failed\n";
    glfwTerminate();
    return 77;
  }
  glfwMakeContextCurrent(window);
  auto &backend = arbitgpu::nativeFixtureSceneBackend();
  CapabilitySource capabilities(backend);
  PortContract geometryContract;
  geometryContract.carrier = CarrierKind::geometry3D;
  geometryContract.overflow = OverflowPolicy::reject;
  geometryContract.maxVertices = 4;
  geometryContract.maxIndices = 6;
  PortContract pointsContract;
  pointsContract.carrier = CarrierKind::points3D;
  pointsContract.overflow = OverflowPolicy::reject;
  pointsContract.maxPoints = 4;
  PortContract contract;
  contract.carrier = CarrierKind::instances3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxInstances = 4;
  std::string error;
  if (!verifyNativeGeometryAdmission(backend,error)) return fail(error) ? 0 : 1;
  auto pointGrid = lowerGrid(10, 11, 1, 2, 2, 2.0f, geometryContract, error);
  auto points = pointGrid
      ? lowerPointsFromVertices(*pointGrid, 12, pointsContract, error)
      : std::nullopt;
  auto sourceGrid = lowerGrid(20, 11, 1, 2, 2, 0.5f, geometryContract, error);
  auto value = points && sourceGrid
      ? lowerInstanceOnPoints(*points, sourceGrid->stableId, 30, contract, error)
      : std::nullopt;
  if (!value) return fail(error) ? 0 : 1;
  value->operations.insert(value->operations.begin(), sourceGrid->operations.begin(),
                           sourceGrid->operations.end());
  value->dispatchCount += sourceGrid->dispatchCount;
  auto admitted = admitValue(*value, contract, error);
  if (!admitted) return fail(error) ? 0 : 1;
  auto bytes = lowerRuntimePlan(contract, *admitted);
  videohelper::geometry::GeometryCorePlanRuntime runtime(capabilities);
  const videohelper::geometry::PlanOwnerIdentity owner{1, 1, 1, 1, 1,
      videohelper::geometry::PlanUse::preview};
  auto plan = runtime.admitPreview(owner, bytes, ResourceLimits{}, {}, error);
  if (!plan) return fail(error) ? 0 : 1;
  auto execution = videohelper::geometry::executeNativeGeometry(
      backend, *plan, 64, 64, error, true);
  if (!execution) return fail(error) ? 0 : 1;
  const std::vector<StableId> collisionIds {4, 255, 5, 5 + (StableId{1} << 24)};
  const std::vector<StableId> reorderedCollisionIds {
      5 + (StableId{1} << 24), 4, 5, 255};
  std::array<std::uint32_t, 4> collisionSlots {};
  std::array<std::uint32_t, 4> reorderedCollisionSlots {};
  for (std::size_t index = 0; index < collisionIds.size(); ++index) {
    collisionSlots[index] = videohelper::geometry::diagnosticColorSlot(
        collisionIds, collisionIds[index]);
    reorderedCollisionSlots[index] = videohelper::geometry::diagnosticColorSlot(
        reorderedCollisionIds, collisionIds[index]);
  }
  const bool collisionOracle = collisionSlots == reorderedCollisionSlots
      && collisionSlots[0] != collisionSlots[1]
      && collisionSlots[2] != collisionSlots[3]
      && collisionSlots[0] != collisionSlots[2]
      && collisionSlots[0] != collisionSlots[3]
      && collisionSlots[1] != collisionSlots[2]
      && collisionSlots[1] != collisionSlots[3]
      && collisionSlots != std::array<std::uint32_t, 4>{
          collisionSlots[1], collisionSlots[0], collisionSlots[3], collisionSlots[2]};
  auto productionExecution = videohelper::geometry::executeNativeGeometry(
      backend, *plan, 64, 64, error);
  if (!productionExecution) return fail(error) ? 0 : 1;
  const auto noteInputs = twoNoteRuntimeInputs();
  auto coexistenceExecution = videohelper::geometry::executeNativeGeometry(
      backend, *plan, 64, 64, error, false, noteInputs);
  if (!coexistenceExecution) return fail(error) ? 0 : 1;
  if (!coexistenceExecution->resources
      || coexistenceExecution->drawnInstanceIds != execution->drawnInstanceIds
      || coexistenceExecution->stats.instancedDrawCount != 2
      || coexistenceExecution->stats.instanceBufferUploadCount != 1
      || coexistenceExecution->stats.submittedInstanceCount != 8
      || coexistenceExecution->stats.noteInstanceDrawCount != 2
      || coexistenceExecution->stats.noteInstanceTransformUploadCount != 1
      || coexistenceExecution->stats.submittedNoteInstanceCount != 8
      || coexistenceExecution->stats.ordinaryDrawCount != 0)
    return fail("Geometry Instances3D and Block C note instancing collided") ? 0 : 1;
  auto reorderedValue = *value;
  auto &reorderedInstances = std::get<InstancesData>(reorderedValue.data).instances;
  std::swap(reorderedInstances[0], reorderedInstances[1]);
  std::swap(reorderedInstances[2], reorderedInstances[3]);
  if (admitValue(reorderedValue, contract, error)
      || error != "retained Instances3D disagrees with its operation result")
    return fail("reordered retained Instances3D did not fail closed against its operation result") ? 0 : 1;
  {
    auto ordinaryScene = ordinarySharedScene();
    auto ordinaryPreparation = backend.prepare(ordinaryScene);
    auto ordinaryRender = ordinaryPreparation.prepared
        ? backend.render(ordinaryScene, ordinaryPreparation.resources, 64, 64)
        : arbitgpu::NativeFixtureSceneSubmission{};
    if (!ordinaryRender.rendered || ordinaryRender.stats.ordinaryDrawCount != 2
        || ordinaryRender.stats.instancedDrawCount != 0
        || ordinaryRender.stats.instanceBufferUploadCount != 0)
      return fail("ordinary shared-geometry Scene3D was routed through instancing") ? 0 : 1;
    auto mismatchedAdmissionPreparation = backend.prepareGeometryInstances(
        ordinaryScene, {}, plan->plan);
    auto mismatchedAdmissionRender = mismatchedAdmissionPreparation.prepared
        ? backend.render(ordinaryScene, mismatchedAdmissionPreparation.resources, 64, 64)
        : arbitgpu::NativeFixtureSceneSubmission{};
    if (!mismatchedAdmissionRender.rendered
        || mismatchedAdmissionRender.stats.ordinaryDrawCount != 2
        || mismatchedAdmissionRender.stats.instancedDrawCount != 0)
      return fail("an admitted Instances3D token authorized a different scene") ? 0 : 1;
  }

  arbitgl::GlFuncs gl;
  std::string missing;
  if (!arbitgl::loadGlFunctions(gl, missing)) return fail(missing) ? 0 : 1;
  GLuint framebuffer = 0;
  gl.GenFramebuffers(1, &framebuffer);
  gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  const auto readPixels = [&](const videohelper::geometry::NativeGeometryExecution &render) {
    gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        static_cast<GLuint>(render.frame->colorImageHandle()), 0);
    std::vector<std::uint8_t> result(64u * 64u * 4u);
    if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
      glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, result.data());
    return result;
  };
  const auto pixels = readPixels(*execution);
  const auto readDistributedPixels = [&](const auto &frame) {
    gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        static_cast<GLuint>(frame->colorImageHandle()), 0);
    std::vector<std::uint8_t> result(64u*64u*4u);
    if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE)
      glReadPixels(0,0,64,64,GL_RGBA,GL_UNSIGNED_BYTE,result.data());
    return result;
  };
  if (!verifyNativeDistributedInstances(backend,readDistributedPixels,error)) return fail(error) ? 0 : 1;
  const auto readScenePixels = [&](const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame) {
    gl.BindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,
        static_cast<GLuint>(frame->colorImageHandle()),0);
    std::vector<std::uint8_t> result(frame->width()*frame->height()*4u);
    if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE)
      glReadPixels(0,0,frame->width(),frame->height(),GL_RGBA,GL_UNSIGNED_BYTE,result.data());
    return result;
  };
  {
    // Reproduce the product's gap between solid frames: no scene-resource or
    // frame lease survives until the next prepare, even at the same revision.
    backend.releaseCachedProgramsForCurrentContext();
    auto parameters = solidBodyFixture();
    std::weak_ptr<const arbitgpu::NativeFixtureSceneResources> previousResources;
    std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame> previousFrame;
    std::vector<std::uint8_t> originalPixels, movedPixels;
    const double times[] = {0.5, 0.9, 0.5};
    for (int index = 0; index < 3; ++index) {
      if (!previousResources.expired() || !previousFrame.expired())
        return fail("program cache retained a previous solid scene or frame") ? 0 : 1;
      const auto state = sampleSolidBodyFixture(parameters, times[index], error);
      const auto scene = videorender::particleSolidScene(parameters, state, 7, error);
      if (!error.empty() || !scene) return fail(error) ? 0 : 1;
      const auto prepared = backend.prepare(scene);
      if (!prepared.prepared || prepared.stats.shaderProgramBuildCount != (index == 0 ? 1u : 0u)
          || prepared.stats.staticUploadCount != 1)
        return fail("solid frames must reuse their program while uploading new scene buffers: "
                    + prepared.error) ? 0 : 1;
      const auto rendered = backend.render(scene, prepared.resources, 96, 96);
      if (!rendered.rendered) return fail(rendered.error) ? 0 : 1;
      const auto pixels = readScenePixels(rendered.frame);
      if (index == 0) originalPixels = pixels;
      else if (index == 1) movedPixels = pixels;
      else if (pixels != originalPixels || movedPixels == originalPixels)
        return fail("program reuse changed solid seek replay or froze the dynamic scene") ? 0 : 1;
      previousResources = prepared.resources;
      previousFrame = rendered.frame;
    }
    if (!previousResources.expired() || !previousFrame.expired())
      return fail("solid scene resources survived their final compositor lease") ? 0 : 1;

    // A separate export context owns its own linked program and can retire its
    // cache without invalidating the viewport's program.
    auto* exportContext = glfwCreateWindow(64, 64, "solid-program-export", nullptr, nullptr);
    if (!exportContext) return fail("could not create isolated program-cache context") ? 0 : 1;
    glfwMakeContextCurrent(exportContext);
    {
      const auto scene = ordinarySharedScene();
      const auto prepared = backend.prepare(scene);
      if (!prepared.prepared || prepared.stats.shaderProgramBuildCount != 1)
        return fail("a second GL context reused the viewport's linked program") ? 0 : 1;
      videohelper::geometry::releaseGeometryExecutionCache();
      if (!backend.render(scene, prepared.resources, 64, 64).rendered)
        return fail("cache release invalidated an outstanding scene-resource lease") ? 0 : 1;
    }
    glfwDestroyWindow(exportContext);
    glfwMakeContextCurrent(window);
    {
      const auto prepared = backend.prepare(ordinarySharedScene());
      if (!prepared.prepared || prepared.stats.shaderProgramBuildCount != 0)
        return fail("export context retirement removed the viewport's program") ? 0 : 1;
    }
    videohelper::geometry::releaseGeometryExecutionCache();
    {
      const auto prepared = backend.prepare(ordinarySharedScene());
      if (!prepared.prepared || prepared.stats.shaderProgramBuildCount != 1)
        return fail("context teardown retained a cached linked program") ? 0 : 1;
    }
  }
  if (!verifyNativeGeometryComposition(backend,readScenePixels,false,error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeConstructedFields(backend,readDistributedPixels,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeImportedGeometry(backend,readDistributedPixels,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeInstanceAppearance(backend,readDistributedPixels,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeInstanceMaterialFields(backend,readDistributedPixels,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeReactiveSurface(backend,readDistributedPixels,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeReactiveFrame(backend,readDistributedPixels,ordinarySharedScene(),error)) return fail(error) ? 0 : 1;
  if (!verifyNativeSolidBodies(readScenePixels,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeSolidBodies(readScenePixels,error,{},true)) return fail(error) ? 0 : 1;
  {
    const auto imported=videohelper::gltf::adaptGlbMeshToGeometryCore(importedGeometryFixture(),1,9001,9000,1,error);
    if (!imported || !verifyNativeSolidBodies(readScenePixels,error,
        std::make_shared<const ValueDescriptor>(imported->descriptor()))) return fail(error) ? 0 : 1;
  }
  if (!verifyNativeObjectSurfacePrograms(backend,readDistributedPixels,ordinarySharedScene(),error)) return fail(error) ? 0 : 1;
  if (!verifyNativeMaterialTables(backend,readDistributedPixels,error)) return fail(error) ? 0 : 1;
  {
    auto meshContract=geometryContract;
    meshContract.maxVertices=64; meshContract.maxIndices=256;
    auto cube=lowerMeshGenerator(2101,2101,1,OperationCode::cube,{0.5f,0.5f,0.5f,0},meshContract,error);
    Transform leftTransform; leftTransform.translation={-0.75f,0,0};
    Transform rightTransform; rightTransform.translation={0.75f,0,0};
    rightTransform.scale={0.75f,1.5f,1}; rightTransform.rotation={0,0,0.38268343f,0.9238795f};
    auto left=cube ? lowerTransformGeometry(*cube,2102,leftTransform,meshContract,error) : std::nullopt;
    auto right=cube ? lowerTransformGeometry(*cube,2103,rightTransform,meshContract,error) : std::nullopt;
    auto joined=left && right ? lowerJoinGeometry(*left,*right,2104,meshContract,error) : std::nullopt;
    auto box=joined ? lowerBoundingBox(*joined,2105,meshContract,error) : std::nullopt;
    if (!cube || !joined || !box) return fail(error) ? 0 : 1;
    videohelper::geometry::GeometryCorePlanRuntime operationRuntime(capabilities);
    const auto renderValue=[&](const ValueDescriptor &mesh,bool exportUse) -> std::optional<std::vector<std::uint8_t>> {
      auto admittedMesh=admitValue(mesh,meshContract,error);
      if (!admittedMesh) return std::nullopt;
      const auto payload=lowerRuntimePlan(meshContract,*admittedMesh);
      const videohelper::geometry::PlanOwnerIdentity identity{1,1,1,mesh.stableId,1,
          exportUse ? videohelper::geometry::PlanUse::exportRender : videohelper::geometry::PlanUse::preview};
      auto admittedPlan=exportUse ? operationRuntime.admitExport(identity,payload,{}, {},error)
                                  : operationRuntime.admitPreview(identity,payload,{}, {},error);
      if (!admittedPlan) return std::nullopt;
      auto draw=videohelper::geometry::executeNativeGeometry(backend,*admittedPlan,64,64,error);
      if (!draw || draw->stats.ordinaryDrawCount!=1) return std::nullopt;
      return readPixels(*draw);
    };
    const auto singlePixels=renderValue(*cube,false);
    const auto joinedPixels=renderValue(*joined,false);
    const auto joinedExport=renderValue(*joined,true);
    const auto boxPixels=renderValue(*box,false);
    const auto boxExport=renderValue(*box,true);
    if (!singlePixels || !joinedPixels || !joinedExport || !boxPixels || !boxExport)
      return fail(error) ? 0 : 1;
    if (*singlePixels==*joinedPixels || *joinedPixels==*boxPixels ||
        *joinedPixels!=*joinedExport || *boxPixels!=*boxExport)
      return fail("TRS join and bounding box must change native pixels and agree in preview and export") ? 0 : 1;
  }
  {
    PortContract fieldContract;
    fieldContract.carrier=CarrierKind::field;
    fieldContract.fieldDomain=Domain::vertex;
    fieldContract.fieldValueType=ValueType::vector;
    fieldContract.fieldInterpolation=Interpolation::constant;
    fieldContract.maxFieldElements=4;
    auto field=lowerElementField(*pointGrid,901,OperationCode::positionField,fieldContract,error);
    auto displaced=field ? lowerVectorDisplacement(*pointGrid,*field,902,0.5f,geometryContract,error)
                         : std::nullopt;
    auto displacedAdmission=displaced ? admitValue(*displaced,geometryContract,error) : std::nullopt;
    auto originalAdmission=admitValue(*pointGrid,geometryContract,error);
    if (!displacedAdmission || !originalAdmission) return fail(error) ? 0 : 1;
    const auto displacedBytes=lowerRuntimePlan(geometryContract,*displacedAdmission);
    auto originalPlan=runtime.admitPreview({1,1,1,901,1,videohelper::geometry::PlanUse::preview},
        lowerRuntimePlan(geometryContract,*originalAdmission),{}, {},error);
    auto previewPlan=runtime.admitPreview({1,1,1,902,1,videohelper::geometry::PlanUse::preview},
        displacedBytes,{}, {},error);
    auto exportPlan=runtime.admitExport({1,1,1,903,1,videohelper::geometry::PlanUse::exportRender},
        displacedBytes,{}, {},error);
    if (!originalPlan || !previewPlan || !exportPlan) return fail(error) ? 0 : 1;
    auto originalDraw=videohelper::geometry::executeNativeGeometry(backend,*originalPlan,64,64,error);
    auto previewDraw=videohelper::geometry::executeNativeGeometry(backend,*previewPlan,64,64,error);
    auto exportDraw=videohelper::geometry::executeNativeGeometry(backend,*exportPlan,64,64,error);
    if (!originalDraw || !previewDraw || !exportDraw) return fail(error) ? 0 : 1;
    const auto before=readPixels(*originalDraw);
    const auto previewPixels=readPixels(*previewDraw);
    const auto exportPixels=readPixels(*exportDraw);
    if (before == previewPixels || previewPixels != exportPixels)
      return fail("element field displacement must change pixels and agree in preview and export") ? 0 : 1;
  }
  const auto productionPixels = readPixels(*productionExecution);
  const auto readNotePixels = [&](const auto& frame) {
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        static_cast<GLuint>(frame->colorImageHandle()), 0);
    std::vector<std::uint8_t> result(64u * 64u * 4u);
    if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
      glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, result.data());
    return result;
  };
  if (!verifyNativeNoteAppearance(backend, readNotePixels, false, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeSpectrumInstancer(backend, readNotePixels, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeSpectralHistory(backend, readNotePixels, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeAudioDeformer(backend, readNotePixels, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeOrderedAudioGeometry(backend, readNotePixels, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeScoreFields(backend, readNotePixels, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeTimelineCurves(backend, readNotePixels, error))
    return fail(error) ? 0 : 1;

  for (int analysisSource=0; analysisSource<3; ++analysisSource) {
    auto spectrumMesh=*sourceGrid;
    spectrumMesh.operations.push_back({901,OperationCode::evaluateField,901,901,{}});
    spectrumMesh.operations.push_back({902,OperationCode::transform,spectrumMesh.stableId,902,{}});
    spectrum::Binding binding;
    binding.source=static_cast<spectrum::Source>(analysisSource);
    binding.sourceTrackId=analysisSource==0 ? -1 : 37;
    binding.fieldStableId=901; binding.sourceGeometryStableId=spectrumMesh.stableId;
    binding.consumerGeometryStableId=902; binding.direction={2.0f,0.0f,0.0f};
    binding.coordinates={0.0f,1.0f,0.0f,1.0f};
    binding.attackSeconds=0.1f; binding.releaseSeconds=0.3f;
    spectrumMesh.spectrumFields.push_back(binding);
    spectrumMesh.stableId=902; spectrumMesh.dispatchCount+=2;
    auto spectrumAdmitted=admitValue(spectrumMesh,geometryContract,error);
    if (!spectrumAdmitted) return fail(error) ? 0 : 1;
    videohelper::geometry::GeometryCorePlanRuntime spectrumRuntime(capabilities);
    const auto spectrumBytes=lowerRuntimePlan(geometryContract,*spectrumAdmitted);
    auto previewPlan=spectrumRuntime.admitPreview({1,1,1,902,1,videohelper::geometry::PlanUse::preview},
                                                   spectrumBytes,{}, {},error);
    auto exportPlan=spectrumRuntime.admitExport({1,1,1,902,1,videohelper::geometry::PlanUse::exportRender},
                                                 spectrumBytes,{}, {},error);
    if (!previewPlan || !exportPlan) return fail(error) ? 0 : 1;
    videohelper::geometry::SpectrumEvaluation input;
    input.featuresAt=[](double seconds) {
      spectrum::Bands bands {}; bands[63]=seconds>=0.5 ? 0.8f : 0.0f; return bands;
    };
    input.sourceFeaturesAt=[source=binding.source, sample=input.featuresAt](spectrum::Source requested,
        std::int32_t track,double seconds,spectrum::Bands& bands) {
      if (requested!=source || track!=37) return false;
      bands=sample(seconds); return true;
    };
    input.timeSeconds=0.2; input.bands=input.featuresAt(input.timeSeconds);
    auto silent=videohelper::geometry::executeNativeGeometry(backend,*previewPlan,64,64,error,false,{},&input);
    input.timeSeconds=0.8; input.bands=input.featuresAt(input.timeSeconds);
    auto active=videohelper::geometry::executeNativeGeometry(backend,*previewPlan,64,64,error,false,{},&input);
    auto exportedInput=input; exportedInput.followers.clear();
    auto exported=videohelper::geometry::executeNativeGeometry(backend,*exportPlan,64,64,error,false,{},&exportedInput);
    if (!silent || !active || !exported) return fail(error) ? 0 : 1;
    if (readPixels(*active)==readPixels(*silent) || readPixels(*active)!=readPixels(*exported)
        || std::get<GeometryData>(spectrumMesh.data).positions.size()!=4)
      return fail("spectrum displacement must change native pixels and match preview/export without expanding topology") ? 0 : 1;
    input.timeSeconds=0.2; input.bands=input.featuresAt(input.timeSeconds);
    auto sought=videohelper::geometry::executeNativeGeometry(backend,*previewPlan,64,64,error,false,{},&input);
    if (!sought || readPixels(*sought)!=readPixels(*silent))
      return fail("spectrum seek did not restore the fixed-frame native result") ? 0 : 1;
    if (analysisSource!=0) {
      input.sourceFeaturesAt={};
      if (videohelper::geometry::executeNativeGeometry(backend,*previewPlan,64,64,error,false,{},&input)
          || error.find("Selected track/group spectrum is unavailable")==std::string::npos)
        return fail("missing selected source must not silently fall back to master") ? 0 : 1;
    }
  }

  for (const auto &generator : std::array<std::pair<OperationCode, std::array<float, 4>>, 5>{{
      {OperationCode::plane,{2,2,0,0}}, {OperationCode::cube,{2,2,2,0}},
      {OperationCode::sphere,{1,32,16,0}}, {OperationCode::icosphere,{1,2,0,0}},
      {OperationCode::cylinder,{1,2,32,0}}}}) {
    auto meshContract = geometryContract;
    meshContract.maxVertices = 4096;
    meshContract.maxIndices = 12288;
    auto generated = lowerMeshGenerator(701,701,1,generator.first,generator.second,
                                        meshContract,error);
    auto generatedAdmission = generated ? admitValue(*generated,meshContract,error) : std::nullopt;
    if (!generatedAdmission) return fail(error) ? 0 : 1;
    const auto generatedBytes = lowerRuntimePlan(meshContract,*generatedAdmission);
    videohelper::geometry::GeometryCorePlanRuntime generatedRuntime(capabilities);
    const videohelper::geometry::PlanOwnerIdentity previewOwner{1,1,1,701,1,
        videohelper::geometry::PlanUse::preview};
    const videohelper::geometry::PlanOwnerIdentity exportOwner{1,1,1,701,1,
        videohelper::geometry::PlanUse::exportRender};
    auto previewPlan = generatedRuntime.admitPreview(previewOwner,generatedBytes,{}, {},error);
    auto exportPlan = generatedRuntime.admitExport(exportOwner,generatedBytes,{}, {},error);
    if (!previewPlan || !exportPlan) return fail(error) ? 0 : 1;
    auto previewMesh = videohelper::geometry::executeNativeGeometry(backend,*previewPlan,64,64,error);
    auto exportMesh = videohelper::geometry::executeNativeGeometry(backend,*exportPlan,64,64,error);
    if (!previewMesh || !exportMesh) return fail(error) ? 0 : 1;
    const auto previewImage = readPixels(*previewMesh);
    const auto exportImage = readPixels(*exportMesh);
    if (previewImage != exportImage)
      return fail("generated mesh preview and export pixels differ") ? 0 : 1;
    std::size_t visible = 0;
    for (std::size_t i = 0; i < previewImage.size(); i += 4)
      if (std::abs(int(previewImage[i]) - int(previewImage[0]))
          + std::abs(int(previewImage[i+1]) - int(previewImage[1]))
          + std::abs(int(previewImage[i+2]) - int(previewImage[2])) > 5) ++visible;
    if (visible < 30 || previewMesh->stats.ordinaryDrawCount != 1)
      return fail("generated mesh did not produce a visible native indexed draw") ? 0 : 1;
  }

  for (const bool circle : {false, true}) {
    PortContract curveContract;
    curveContract.carrier = CarrierKind::curves3D;
    curveContract.maxCurvePoints = 32;
    curveContract.maxSplines = 1;
    auto meshContract = geometryContract;
    meshContract.maxVertices = 64;
    meshContract.maxIndices = 192;
    auto curve = circle ? lowerCurveCircle(801,801,1,32,1,curveContract,error)
                        : lowerCurveLine(801,801,1,8,2,curveContract,error);
    auto ribbon = curve ? lowerCurveToMesh(*curve,802,meshContract,error,0.25f) : std::nullopt;
    auto admittedRibbon = ribbon ? admitValue(*ribbon,meshContract,error) : std::nullopt;
    if (!admittedRibbon) return fail(error) ? 0 : 1;
    const auto ribbonBytes = lowerRuntimePlan(meshContract,*admittedRibbon);
    videohelper::geometry::GeometryCorePlanRuntime ribbonRuntime(capabilities);
    const videohelper::geometry::PlanOwnerIdentity previewOwner{1,1,1,802,1,
        videohelper::geometry::PlanUse::preview};
    const videohelper::geometry::PlanOwnerIdentity exportOwner{1,1,1,802,1,
        videohelper::geometry::PlanUse::exportRender};
    auto previewPlan = ribbonRuntime.admitPreview(previewOwner,ribbonBytes,{}, {},error);
    auto exportPlan = ribbonRuntime.admitExport(exportOwner,ribbonBytes,{}, {},error);
    if (!previewPlan || !exportPlan) return fail(error) ? 0 : 1;
    auto previewRibbon = videohelper::geometry::executeNativeGeometry(backend,*previewPlan,64,64,error);
    auto exportRibbon = videohelper::geometry::executeNativeGeometry(backend,*exportPlan,64,64,error);
    if (!previewRibbon || !exportRibbon) return fail(error) ? 0 : 1;
    const auto previewImage = readPixels(*previewRibbon);
    if (previewImage != readPixels(*exportRibbon))
      return fail("curve ribbon preview and export pixels differ") ? 0 : 1;
    std::size_t visible = 0;
    for (std::size_t i = 0; i < previewImage.size(); i += 4)
      if (std::abs(int(previewImage[i]) - int(previewImage[0]))
          + std::abs(int(previewImage[i+1]) - int(previewImage[1]))
          + std::abs(int(previewImage[i+2]) - int(previewImage[2])) > 5) ++visible;
    if (visible < 30 || previewRibbon->stats.ordinaryDrawCount != 1)
      return fail("curve ribbon did not produce a visible native indexed draw") ? 0 : 1;
  }

  for (const bool instanced : {false,true}) {
    PortContract curveContract;
    curveContract.carrier=CarrierKind::curves3D; curveContract.maxCurvePoints=32; curveContract.maxSplines=1;
    auto meshContract=geometryContract;
    meshContract.maxVertices=256; meshContract.maxIndices=1536;
    PortContract pointContract;
    pointContract.carrier=CarrierKind::points3D; pointContract.maxPoints=32;
    PortContract outputContract=meshContract;
    if (instanced) {
      outputContract={}; outputContract.carrier=CarrierKind::instances3D; outputContract.maxInstances=32;
    }
    std::vector<std::uint8_t> priorImage;
    for (const bool changed : {false,true}) {
      auto curve=lowerCurveLine(4801,4801,1,3,2,curveContract,error);
      if (!curve) return fail(error) ? 0 : 1;
      for (const auto &step : std::vector<std::pair<OperationCode,std::array<float,4>>>{
          {OperationCode::curveResample,{32,0,0,0}},
          {OperationCode::curveTrim,{0.05f,0.95f,0,0}},
          {OperationCode::curveSetRadius,{0.6f,1.4f,0,0}},
          {OperationCode::curveSetTilt,{0,0.4f,0,0}},
          {OperationCode::curveWave,{0.3f,2,changed ? 0.25f : 0,1}}}) {
        curve=lowerCurveOperation(*curve,curve->stableId+1,step.first,step.second,curveContract,error);
        if (!curve) return fail(error) ? 0 : 1;
      }
      auto value=lowerCurveOperation(*curve,4810,
          instanced ? OperationCode::pointsFromCurves : OperationCode::curveToTube,
          instanced ? std::array<float,4>{0,0.1f,0,0} : std::array<float,4>{0.08f,8,0,0},
          instanced ? pointContract : meshContract,error);
      if (!value) return fail(error) ? 0 : 1;
      if (instanced) {
        auto source=lowerMeshGenerator(4811,4811,1,OperationCode::cube,{1,1,1,0},meshContract,error);
        value=lowerInstanceOnPoints(*value,4811,4812,outputContract,error);
        if (!source || !value) return fail(error) ? 0 : 1;
        auto operations=source->operations;
        if (!appendUniqueOperations(operations,value->operations,error)) return fail(error) ? 0 : 1;
        value->operations=std::move(operations); value->dispatchCount=value->operations.size();
      }
      auto admitted=admitValue(*value,outputContract,error);
      if (!admitted) return fail(error) ? 0 : 1;
      const auto bytes=lowerRuntimePlan(outputContract,*admitted);
      videohelper::geometry::GeometryCorePlanRuntime curveRuntime(capabilities);
      const videohelper::geometry::PlanOwnerIdentity previewOwner{1,1,1,4812,changed ? 2u : 1u,
          videohelper::geometry::PlanUse::preview};
      auto exportOwner=previewOwner; exportOwner.use=videohelper::geometry::PlanUse::exportRender;
      auto previewPlan=curveRuntime.admitPreview(previewOwner,bytes,{}, {},error);
      auto exportPlan=curveRuntime.admitExport(exportOwner,bytes,{}, {},error);
      if (!previewPlan || !exportPlan) return fail(error) ? 0 : 1;
      auto preview=videohelper::geometry::executeNativeGeometry(backend,*previewPlan,64,64,error);
      auto exported=videohelper::geometry::executeNativeGeometry(backend,*exportPlan,64,64,error);
      if (!preview || !exported) return fail(error) ? 0 : 1;
      const auto pixels=readPixels(*preview);
      if (pixels!=readPixels(*exported)) return fail("edited curve preview and export pixels differ") ? 0 : 1;
      if (changed && pixels==priorImage) return fail("curve phase edit does not change native pixels") ? 0 : 1;
      std::size_t visible=0;
      for (std::size_t i=0;i<pixels.size();i+=4)
        if (std::abs(int(pixels[i])-int(pixels[0]))+std::abs(int(pixels[i+1])-int(pixels[1]))+
            std::abs(int(pixels[i+2])-int(pixels[2]))>5) ++visible;
      if (visible<30 || (instanced && (preview->stats.instancedDrawCount!=1 || preview->stats.submittedInstanceCount!=32)) ||
          (!instanced && preview->stats.ordinaryDrawCount!=1))
        return fail("edited curve did not produce visible tube or instance output") ? 0 : 1;
      priorImage=pixels;
    }
  }

  {
    struct HarmonicLayer {
      std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
      unsigned texture = 0;
      float opacity = 1;
      std::string nativeTextureBackend;
      std::uintptr_t nativeTextureView = 0;
      arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
      std::shared_ptr<const void> nativeTextureOwner;
      int texWidth = 0, texHeight = 0;
    } previewLayer, exportLayer;
    auto score = std::make_shared<arbitmod::Score>();
    score->scoreRevision = 1;
    score->rootFreq = 440;
    for (int index = 0; index < 2; ++index) {
      arbitmod::Note note;
      note.id = -index - 1;
      note.startBeat = static_cast<float>(index) - 0.5f;
      note.lengthBeats = 4;
      note.freqHz = index == 0 ? 440 : 660;
      note.velocity = 100;
      note.durationSeconds = 2;
      score->notes.push_back(note);
    }
    score->links.push_back({-41, -2, -1, 3, 2, 0});
    canonicalblockc::FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
    key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
    key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
    key.fps = 60;
    canonicalblockc::FrameProducer previewProducer, exportProducer;
    previewLayer.canonicalBlockCFrame = previewProducer.evaluate(key, score, 0);
    exportLayer.canonicalBlockCFrame = exportProducer.evaluate(key, score, 0);
    visualharmonicgeometry::Mapping mapping;
    mapping.stableId = 851;
    mapping.notes.meshScale = 0.3f;
    const videohelper::geometry::PlanOwnerIdentity previewOwner{1,1,1,851,1,
        videohelper::geometry::PlanUse::preview};
    const videohelper::geometry::PlanOwnerIdentity exportOwner{1,1,1,851,1,
        videohelper::geometry::PlanUse::exportRender};
    if (!previewLayer.canonicalBlockCFrame || !exportLayer.canonicalBlockCFrame
        || !videohelper::harmonicgeometry::render(mapping, previewOwner, 64, 64, previewLayer, error)
        || !videohelper::harmonicgeometry::render(mapping, exportOwner, 64, 64, exportLayer, error))
      return fail("harmonic ribbon production render failed: " + error) ? 0 : 1;
    const auto preview = std::static_pointer_cast<const videohelper::harmonicgeometry::RenderedFrame>(
        previewLayer.nativeTextureOwner);
    const auto exported = std::static_pointer_cast<const videohelper::harmonicgeometry::RenderedFrame>(
        exportLayer.nativeTextureOwner);
    const auto previewImage = readPixels(preview->native);
    if (previewImage != readPixels(exported->native) || preview->native.stats.ordinaryDrawCount != 1
        || preview->admittedGeometry->value().descriptor().attributes[0].elements[0].components[0] != -41)
      return fail("harmonic ribbon preview/export pixels or native draw disagree") ? 0 : 1;
    std::size_t visible = 0;
    for (std::size_t i = 0; i < previewImage.size(); i += 4)
      if (std::abs(int(previewImage[i]) - int(previewImage[0]))
          + std::abs(int(previewImage[i+1]) - int(previewImage[1]))
          + std::abs(int(previewImage[i+2]) - int(previewImage[2])) > 5) ++visible;
    if (visible < 12)
      return fail("harmonic ribbon has no visible native pixels") ? 0 : 1;
    ++key.frame;
    key.beat = 0.5;
    previewLayer.canonicalBlockCFrame = previewProducer.evaluate(key, score, 0.5f);
    if (!videohelper::harmonicgeometry::render(mapping, previewOwner, 64, 64, previewLayer, error))
      return fail("harmonic ribbon next frame could not coexist with retained prior frame") ? 0 : 1;
    const auto next = std::static_pointer_cast<const videohelper::harmonicgeometry::RenderedFrame>(
        previewLayer.nativeTextureOwner);
    if (readPixels(next->native) == previewImage || readPixels(preview->native) != previewImage)
      return fail("harmonic ribbon time did not move or overwrote a retained frame") ? 0 : 1;
  }

  const auto pixel = [&](unsigned x, unsigned y, unsigned channel) {
    return pixels[(y * 64u + x) * 4u + channel];
  };
  const auto colorAt = [](const std::vector<std::uint8_t> &image,
                          unsigned x, unsigned y) {
    const auto offset = (y * 64u + x) * 4u;
    return std::array<std::uint8_t, 3>{image[offset], image[offset + 1], image[offset + 2]};
  };
  std::size_t changedPixels = 0;
  for (unsigned y = 0; y < 64; ++y)
    for (unsigned x = 0; x < 64; ++x) {
      const auto difference =
          std::abs(static_cast<int>(pixel(x, y, 0)) - static_cast<int>(pixel(1, 1, 0))) +
          std::abs(static_cast<int>(pixel(x, y, 1)) - static_cast<int>(pixel(1, 1, 1))) +
          std::abs(static_cast<int>(pixel(x, y, 2)) - static_cast<int>(pixel(1, 1, 2)));
      if (difference > 5)
        ++changedPixels;
    }
  const std::array<std::array<unsigned, 2>, 4> probes {{{16, 16}, {47, 16},
                                                        {16, 47}, {47, 47}}};
  std::array<std::array<std::uint8_t, 3>, 4> identityColors {};
  for (std::size_t index = 0; index < probes.size(); ++index)
    identityColors[index] = colorAt(pixels, probes[index][0], probes[index][1]);
  const bool colorsAreDistinct = identityColors[0] != identityColors[1]
      && identityColors[0] != identityColors[2]
      && identityColors[0] != identityColors[3]
      && identityColors[1] != identityColors[2]
      && identityColors[1] != identityColors[3]
      && identityColors[2] != identityColors[3];
  const bool oracle = collisionOracle && changedPixels >= 16 && changedPixels < 1024 &&
      colorsAreDistinct &&
      productionPixels != pixels &&
      !(productionPixels[(16u * 64u + 16u) * 4u] == 255
          && productionPixels[(16u * 64u + 16u) * 4u + 1u] == 0
          && productionPixels[(16u * 64u + 16u) * 4u + 2u] == 0) &&
      execution->stats.drawCount == 1 &&
      execution->stats.instancedDrawCount == 1 &&
      execution->stats.ordinaryDrawCount == 0 &&
      execution->stats.instanceBufferUploadCount == 1 &&
      execution->stats.submittedInstanceCount == 4 &&
      execution->stats.staticUploadCount == 1 &&
      execution->drawnInstanceIds == std::vector<StableId>({1, 2, 3, 4}) &&
      execution->receipt == plan->plan->receipt().backendBinding &&
      runtime.admissionsInGeneration() == 1;
  const auto oldRendererGeneration = execution->frame->colorTextureDescriptor().rendererGeneration;
  std::weak_ptr<const arbitgpu::NativeFixtureSceneResources> retiredResources = execution->resources;
  gl.DeleteFramebuffers(1, &framebuffer);
  execution.reset();
  productionExecution.reset();
  coexistenceExecution.reset();
  plan.reset();
  if (!runtime.reconcileGeneration(2, 1, 1, error) || !retiredResources.expired()
      || runtime.admissionsInGeneration() != 0)
    return fail("generation replacement did not retire lease-free GL resources and reset counters") ? 0 : 1;
  const videohelper::geometry::PlanOwnerIdentity replacementOwner{2, 1, 1, 1, 1,
      videohelper::geometry::PlanUse::preview};
  auto replacementPlan = runtime.admitPreview(
      replacementOwner, bytes, ResourceLimits{}, {}, error);
  auto replacementExecution = replacementPlan
      ? videohelper::geometry::executeNativeGeometry(
            backend, *replacementPlan, 64, 64, error)
      : std::nullopt;
  if (!replacementExecution || runtime.admissionsInGeneration() != 1
      || replacementExecution->frame->colorTextureDescriptor().rendererGeneration
          <= oldRendererGeneration)
    return fail("replacement generation did not allocate a new native resource lifetime") ? 0 : 1;
  replacementExecution.reset();
  replacementPlan.reset();
  if (!runtime.reconcileGeneration(3, 1, 1, error))
    return fail(error) ? 0 : 1;

  // A preview close and an export/probe teardown must retire both retained
  // frames and admission-owned GL resources while their context still exists.
  // Reopening with the same project/clip revision must allocate a new lifetime.
  std::uint64_t previousGeneration = 0;
  videohelper::geometry::releaseGeometryExecutionCache();
  for (int cycle = 0; cycle < 2; ++cycle) {
    {
      const auto prepared = backend.prepare(ordinarySharedScene());
      if (!prepared.prepared || prepared.stats.shaderProgramBuildCount != 1)
        return fail("reopened context acquired a retired program cache entry") ? 0 : 1;
    }
    auto& cache = videohelper::geometry::geometryExecutionCache();
    if (!cache.runtime.reconcileGeneration(1, 1, 1, error))
      return fail(error) ? 0 : 1;
    auto cachedPlan = cache.runtime.admitPreview(owner, bytes, ResourceLimits{}, {}, error);
    auto cachedDraw = cachedPlan
        ? videohelper::geometry::executeNativeGeometry(backend, *cachedPlan, 64, 64, error)
        : std::nullopt;
    if (!cachedDraw || cachedDraw->frame->colorTextureDescriptor().deviceOrContextIdentity
          != reinterpret_cast<std::uintptr_t>(window))
      return fail("reopened geometry did not bind its current GL context: " + error) ? 0 : 1;
    const auto generation = cachedDraw->frame->colorTextureDescriptor().rendererGeneration;
    if (generation <= previousGeneration)
      return fail("reopened context reused a retired geometry resource lifetime") ? 0 : 1;
    previousGeneration = generation;
    std::weak_ptr<const arbitgpu::NativeFixtureSceneResources> resources = cachedDraw->resources;
    std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame> frame = cachedDraw->frame;
    auto& entry = cache.retained.owners[owner];
    entry.admittedPlan = cachedPlan->plan;
    entry.execution = std::make_shared<const videohelper::geometry::NativeGeometryExecution>(
        std::move(*cachedDraw));
    cachedDraw.reset();
    cachedPlan.reset();
    bool independentThreadIsEmpty = false;
    std::thread other([&] {
      independentThreadIsEmpty = !videohelper::geometry::geometryExecutionCacheStorage();
      videohelper::geometry::releaseGeometryExecutionCache();
    });
    other.join();
    if (!independentThreadIsEmpty || resources.expired())
      return fail("export-thread geometry ownership interfered with preview resources") ? 0 : 1;
    videohelper::geometry::releaseGeometryExecutionCache();
    if (!resources.expired() || !frame.expired())
      return fail("geometry resources survived their owning context teardown") ? 0 : 1;
    if (cycle == 0) {
      glfwDestroyWindow(window);
      window = glfwCreateWindow(64, 64, "geometry-core-reopened", nullptr, nullptr);
      if (!window) return fail("could not reopen native geometry context") ? 0 : 1;
      glfwMakeContextCurrent(window);
    }
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  return oracle ? 0 : (fail("native instance pixels or retained identities disagree, changed pixels="
      + std::to_string(changedPixels)) ? 0 : 1);
}
