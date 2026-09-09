#include "../src/geometry_core_backend.h"
#include "../src/geometry_core_diagnostic_colors.h"
#include "../src/canonical_block_c_frame.h"
#include "../src/gl_loader.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
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
  if (!glfwInit()) return fail("GLFW initialization failed") ? 0 : 1;
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  auto *window = glfwCreateWindow(64, 64, "geometry-core-pixel", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return fail("OpenGL 3.3 core context creation failed") ? 0 : 1;
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
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        static_cast<GLuint>(render.frame->colorImageHandle()), 0);
    std::vector<std::uint8_t> result(64u * 64u * 4u);
    if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
      glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, result.data());
    return result;
  };
  const auto pixels = readPixels(*execution);
  const auto productionPixels = readPixels(*productionExecution);

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
  glfwDestroyWindow(window);
  glfwTerminate();
  return oracle ? 0 : (fail("native instance pixels or retained identities disagree, changed pixels="
      + std::to_string(changedPixels)) ? 0 : 1);
}
