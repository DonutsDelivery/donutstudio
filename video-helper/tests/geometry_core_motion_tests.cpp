#include "geometry_core_motion.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Geometry Core keeps the sample each draw rendered and hands it to the next
// presentation frame as its predecessor, so Motion and motion blur see the
// prior spectrum, control bank and positions. A seek, a rate change or a resize
// starts at zero motion. A runtime value that reaches a Vertex Control
// Parameter read marks the draw as changing every frame and fills the bank the
// draw uploads. Instances keep their transforms by identity, so each instance
// moves from where it was, and one born this frame holds still. Each draw reads
// its prior model matrix in full, so a turn under non-uniform scale moves right
// and instanced draws stay one draw.
namespace {
using videowire::VertexModifierOperation;
namespace control = videowire::vertexcontrol;
namespace geometry = videohelper::geometry;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

videowire::VertexModifierRecord record(videowire::VertexModifierStableId id, VertexModifierOperation operation,
                                       std::initializer_list<videowire::VertexModifierStableId> inputs,
                                       std::initializer_list<double> parameters) {
  videowire::VertexModifierRecord result;
  result.stableId = id; result.operation = operation;
  result.resultType = videowire::vertexModifierOperationSchema(operation).resultType;
  for (const auto input : inputs) result.inputs[result.inputCount++] = input;
  for (const auto parameter : parameters) result.parameters[result.parameterCount++] = parameter;
  return result;
}

// Position plus (c, c, c), where c reads control slot 2.
videowire::VertexModifierIr program() {
  videowire::VertexModifierIr ir;
  ir.rootId = 3;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(7, VertexModifierOperation::controlParameter, {}, {2, 0.5}),
                record(4, VertexModifierOperation::vec3Compose, {7, 7, 7}, {}),
                record(3, VertexModifierOperation::boundedDisplacementOutput, {1, 4}, {10})};
  return ir;
}

geometry::GeometryMotionRecord recorded(std::int64_t frame) {
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.timeSeconds = 1.25f;
  inputs.vertexSpectrum[3] = 0.5f;
  inputs.vertexControls = {0, 0, 0.75f};
  auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
  scene->vertices[0].position = {1, 2, 3};
  scene->vertices[1].position = {-4, 5, 0.5f};
  scene->vertexCount = 2;
  return geometry::geometryMotionRecord(inputs, *scene, {frame, 60, 1}, 640, 360);
}

void records() {
  const auto entry = recorded(10);
  check(entry.sample.timeSeconds == 1.25f && entry.sample.vertexSpectrum[3] == 0.5f,
        "the record keeps the draw's time and spectrum");
  check(entry.sample.vertexControls == std::vector<float>({0, 0, 0.75f}), "the record keeps the draw's control bank");
  check(entry.sample.scenePositions && entry.sample.scenePositions->size() == 2
            && (*entry.sample.scenePositions)[1].x == -4.0f && (*entry.sample.scenePositions)[1].z == 0.5f,
        "the record keeps the positions before the vertex program");
}

void predecessors() {
  const std::optional<geometry::GeometryMotionRecord> previous = recorded(10);
  const auto next = geometry::geometryMotionPredecessor(previous, {11, 60, 1}, 640, 360);
  check(next && next->vertexControls.size() == 3 && next->timeSeconds == 1.25f,
        "the next frame reads the recorded sample");
  check(!geometry::geometryMotionPredecessor(previous, {12, 60, 1}, 640, 360), "a skipped frame starts at zero motion");
  check(!geometry::geometryMotionPredecessor(previous, {10, 60, 1}, 640, 360), "the same frame has no predecessor");
  check(!geometry::geometryMotionPredecessor(previous, {9, 60, 1}, 640, 360), "a backward seek starts at zero motion");
  check(!geometry::geometryMotionPredecessor(previous, {11, 30, 1}, 640, 360), "a rate change starts at zero motion");
  check(!geometry::geometryMotionPredecessor(previous, {11, 60, 1}, 1280, 360), "a resize starts at zero motion");
  check(!geometry::geometryMotionPredecessor(previous, {11, 0, 1}, 640, 360), "a frame without a rate has no predecessor");
  check(!geometry::geometryMotionPredecessor(std::nullopt, {11, 60, 1}, 640, 360), "the first frame has no predecessor");
}

std::unique_ptr<HarmonicMIDI::grid::Visual3DScene> instancedScene(float shift) {
  auto scene = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
  scene->vertices[0].position = {0, 0, 0};
  scene->vertexCount = 1;
  const std::uint32_t ids[] = {30, 10, 20};
  for (std::size_t i = 0; i < 3; ++i) {
    auto& object = scene->objects[i];
    object.id = {ids[i]};
    object.vertexCount = 1;
    object.transform.translation = {static_cast<float>(i) + shift, 0, 0};
    object.transform.scale = {2, 2, 2};
  }
  scene->objectCount = 3;
  return scene;
}

void instances() {
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.imageOutput = renderpassoutput::Output::Motion;
  const auto before = instancedScene(0);
  std::optional<geometry::GeometryMotionRecord> previous =
      geometry::geometryMotionRecord(inputs, *before, {4, 24, 1}, 64, 64, true);
  const auto& kept = previous->instanceTransforms;
  check(kept.size() == 3 && kept[0].first == 10 && kept[1].first == 20 && kept[2].first == 30
            && kept[0].second.translation.x == 1.0f,
        "an instanced record keeps each instance's transform, sorted by identity");
  check(geometry::geometryMotionRecord(inputs, *before, {4, 24, 1}, 64, 64).instanceTransforms.empty(),
        "a single mesh keeps no instance transforms");
  check(!geometry::geometryMotionPredecessorRecord(previous, {6, 24, 1}, 64, 64)
            && geometry::geometryMotionPredecessorRecord(previous, {5, 24, 1}, 64, 64),
        "the instance record follows the same predecessor rule as the sample");

  // Next frame: every instance moved +0.5 on x, instance 30 left and 40 was born.
  auto after = instancedScene(0.5f);
  after->objects[0].id = {40};
  const auto predecessor = geometry::geometryMotionPredecessorRecord(previous, {5, 24, 1}, 64, 64);
  geometry::geometryPriorInstanceTransforms(*after, &*predecessor);
  check(after->previousObjectTransformCount == 3, "every instance carries a prior transform");
  check(after->previousObjectTransforms[1].translation.x == 1.0f
            && after->previousObjectTransforms[2].translation.x == 2.0f,
        "an instance's prior is its own transform one frame earlier, matched by identity");
  check(after->previousObjectTransforms[0].translation.x == 0.5f, "an instance born this frame holds still");
  check(!arbitgpu::sceneObjectHasOwnPrior(*after, 0) && arbitgpu::sceneObjectHasOwnPrior(*after, 1)
            && arbitgpu::sceneObjectHasOwnPrior(*after, 2),
        "moved instances move by their own step in Motion");
  // The shared draw reads each instance's prior world from its record.
  const auto prior = arbitgpu::nativePreviousSceneWorld(*after, 1);
  check(prior[0] == 2.0f && prior[5] == 2.0f && prior[12] == 1.0f,
        "the shared draw's record carries the instance's prior world");

  auto first = instancedScene(0);
  geometry::geometryPriorInstanceTransforms(*first, nullptr);
  check(first->previousObjectTransformCount == 3 && !arbitgpu::sceneObjectHasOwnPrior(*first, 0)
            && !arbitgpu::sceneObjectHasOwnPrior(*first, 2),
        "without a predecessor every instance holds still");
}

// Applies a column-major matrix to a point, as the shader does.
std::array<float, 3> apply(const std::array<float, 16>& m, float x, float y, float z) {
  return {m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13],
          m[2] * x + m[6] * y + m[10] * z + m[14]};
}

bool near(const std::array<float, 3>& a, float x, float y, float z) {
  return std::abs(a[0] - x) < 1e-5f && std::abs(a[1] - y) < 1e-5f && std::abs(a[2] - z) < 1e-5f;
}

// One object under a parent that moves it up by 3. The object turns 90 degrees
// about z and moves +1 on x between frames.
std::unique_ptr<HarmonicMIDI::grid::Visual3DScene> turningScene(HarmonicMIDI::grid::SceneVec3 scale) {
  auto scene = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
  scene->objects[0].id = {5};
  scene->objects[0].transform.translation = {0, 3, 0};
  auto& object = scene->objects[1];
  object.id = {6};
  object.parent = {5};
  object.transform.translation = {1, 0, 0};
  object.transform.rotation = {0, 0, std::sqrt(0.5f), std::sqrt(0.5f)};
  object.transform.scale = scale;
  scene->objectCount = 2;
  scene->previousObjectTransformCount = 2;
  scene->previousObjectTransforms[0] = scene->objects[0].transform;
  scene->previousObjectTransforms[1].scale = scale;
  return scene;
}

void priorMatrices() {
  const auto uniform = turningScene({2, 2, 2});
  const auto stretched = turningScene({2, 1, 1});
  check(arbitgpu::sceneObjectStepIsSimilarity(*uniform, 1), "a turn under uniform scale is a similarity");
  check(!arbitgpu::sceneObjectStepIsSimilarity(*stretched, 1), "a turn under non-uniform scale is not a similarity");

  const auto prior = arbitgpu::nativePreviousSceneWorld(*stretched, 1);
  check(near(apply(prior, 1, 1, 0), 2, 4, 0), "the prior world is the parent's world times the prior transform");

  const std::array<float, 16> identity {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::array<float, 16> current = identity;
  current[0] = 7.0f;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.imageOutput = renderpassoutput::Output::Motion;
  check(arbitgpu::nativePreviousObjectMatrix(*uniform, 1, inputs, identity, current) == current,
        "a similarity step keeps the current matrix and travels in the uniforms");
  check(arbitgpu::nativePreviousObjectMatrix(*stretched, 1, inputs, identity, current) == prior,
        "a non-uniform turn carries the full prior world");
  std::array<float, 16> runtime = identity;
  runtime[14] = 4.0f;
  check(near(apply(arbitgpu::nativePreviousObjectMatrix(*stretched, 1, inputs, runtime, current), 1, 1, 0), 2, 4, 4),
        "the prior world takes the runtime transform the draw applies now");
  check(arbitgpu::nativePreviousObjectMatrix(*stretched, 0, inputs, identity, current) == current,
        "an object without its own prior keeps the current matrix");

  arbitgpu::NativeSceneMotionUniforms motion;
  motion.rotation = {0, 0, 1, 0};
  motion.translationScale = {1, 2, 3, 2};
  const auto frame = arbitgpu::sceneFrameMotionUniforms(motion, true);
  check(frame.rotation == motion.rotation && frame.translationScale == motion.translationScale,
        "a targeted draw keeps the frame step");
  const auto untargeted = arbitgpu::sceneFrameMotionUniforms(motion, false);
  check(untargeted.rotation == std::array<float, 4>({0, 0, 0, 1})
            && untargeted.translationScale == std::array<float, 4>({0, 0, 0, 1}),
        "an untargeted draw drops the object step");
  const auto own = arbitgpu::sceneObjectMotionUniforms(motion, *stretched, 1, {}, true);
  check(own.rotation == motion.rotation && own.translationScale == motion.translationScale,
        "a non-uniform step leaves the uniforms at the frame step");
  const auto similar = arbitgpu::sceneObjectMotionUniforms(motion, *uniform, 1, {}, true);
  check(similar.translationScale != motion.translationScale, "a similarity step folds into the uniforms");

  // The instance record packs the three affine columns into three vectors, and
  // the shader rebuilds the matrix from them.
  const auto packed = arbitgpu::nativePreviousMatrixColumns(prior);
  const std::array<float, 16> rebuilt {packed[0][0], packed[0][1], packed[0][2], 0,
                                       packed[0][3], packed[1][0], packed[1][1], 0,
                                       packed[1][2], packed[1][3], packed[2][0], 0,
                                       packed[2][1], packed[2][2], packed[2][3], 1};
  check(rebuilt == prior, "the packed prior columns rebuild the prior world");

  // An instance of the instanced object moves from its own prior placement.
  auto scene = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
  scene->objectInstanceCount = 2;
  scene->objectInstances[1].translation = {5, 0, 0};
  scene->objectInstances[1].scale = {1, 3, 1};
  scene->previousObjectInstanceCount = 2;
  scene->previousObjectInstances[1].scale = {1, 3, 1};
  const auto columns = arbitgpu::nativeObjectInstancePreviousColumns(identity, *scene, 1);
  check(columns[0][0] == 1.0f && columns[1][0] == 3.0f && columns[2][1] == 0.0f,
        "an instance's prior record is its prior placement under non-uniform scale");
}

// A shared draw folds the runtime frame step into each targeted instance's
// prior world; an instance the runtime transform skips keeps none.
void sharedRuntimeStep() {
  arbitgpu::NativeSceneMotionUniforms motion;
  motion.rotation = {0, 0, std::sqrt(0.5f), std::sqrt(0.5f)};  // 90 degrees about z
  motion.translationScale = {0, 0, 1, 2};
  const auto step = arbitgpu::sceneFrameStepMatrix(motion);
  check(near(apply(step, 1, 0, 0), 0, 2, 1), "the frame step matrix turns, scales and moves as the shader does");

  auto scene = instancedScene(0);
  scene->previousObjectTransformCount = scene->objectCount;
  for (std::size_t i = 0; i < scene->objectCount; ++i) scene->previousObjectTransforms[i] = scene->objects[i].transform;
  scene->previousObjectTransforms[2].translation.x = 1.0f;  // instance 2 was one unit left
  const std::array<float, 16> identity {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::array<float, 16> runtime = identity;
  runtime[13] = 5.0f;
  std::array<float, 16> current = identity;
  current[12] = 1.0f;

  check(arbitgpu::nativeSharedInstancePreviousWorld(*scene, 1, motion, false, runtime, current) == current,
        "an untargeted instance without its own prior repeats its current world");
  check(near(apply(arbitgpu::nativeSharedInstancePreviousWorld(*scene, 1, motion, true, runtime, current), 0, 0, 0),
             0, 2, 1),
        "a targeted instance takes the runtime frame step");
  const auto untargeted = arbitgpu::nativeSharedInstancePreviousWorld(*scene, 2, motion, false, runtime, current);
  check(near(apply(untargeted, 0, 0, 0), 1, 0, 0), "an untargeted instance moves by its own prior alone");
  const auto targeted = arbitgpu::nativeSharedInstancePreviousWorld(*scene, 2, motion, true, runtime, current);
  check(near(apply(targeted, 0, 0, 0), -10, 2, 1),
        "a targeted instance takes its prior under the runtime transform, then the frame step");
}

void runtimeMap() {
  geometrysurfacematerial::ProgramSet programs(2);
  programs[1].material.vertexModifier = program();
  const std::map<std::string, double> unrelated {{"visual9/value", 1.0}};
  const std::map<std::string, double> driven {{control::runtimeKey(7), -3.0}};
  check(!geometry::geometryControlsModulated(programs, nullptr), "no runtime map leaves the draw static");
  check(!geometry::geometryControlsModulated(programs, &unrelated), "a value for another node leaves the draw static");
  check(geometry::geometryControlsModulated(programs, &driven), "a value for the control read changes every frame");
  programs[1].material.vertexModifier->records[1].operation = VertexModifierOperation::importedPosition;
  check(!geometry::geometryControlsModulated(programs, &driven), "a program without a control read is static");

  std::vector<float> values;
  std::string error;
  check(control::bank(program(), nullptr, values, error) && values.size() == 3 && values[2] == 0.5f,
        "without a runtime value the draw uploads the authored value");
  check(control::bank(program(), &driven, values, error) && values[2] == -3.0f,
        "the runtime value reaches the uploaded bank");
}
} // namespace

int main() {
  records();
  predecessors();
  instances();
  priorMatrices();
  sharedRuntimeStep();
  runtimeMap();
  if (failures != 0) return 1;
  std::puts("geometry core motion tests passed");
  return 0;
}
