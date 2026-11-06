#include "gpu_backend/fixture_vertex_modifier_shader.h"
#include "vertex_program_motion.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// Path Trace's prior vertex positions: the vertex program runs on the prior
// frame's scene positions with the prior frame's time, spectrum and control
// bank. Without a prior sample the prior frame is this frame.
namespace {
using videowire::VertexModifierOperation;
using videohelper::materialprogram::BackendTarget;
namespace cpu = videohelper::vertexprogramcpu;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

bool near(const HarmonicMIDI::grid::SceneVec3& a, float x, float y, float z) {
  return std::abs(a.x - x) < 1e-6f && std::abs(a.y - y) < 1e-6f && std::abs(a.z - z) < 1e-6f;
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

// Position plus (c, c, c), where c reads control slot 0.
std::shared_ptr<const videohelper::materialprogram::MaterialProgramRecord> compiled(std::string& error) {
  videowire::VertexModifierIr ir;
  ir.rootId = 3;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(2, VertexModifierOperation::controlParameter, {}, {0, 0.0}),
                record(4, VertexModifierOperation::vec3Compose, {2, 2, 2}, {}),
                record(3, VertexModifierOperation::boundedDisplacementOutput, {1, 4}, {10})};
  const auto admitted = videohelper::vertexmodifier::admitVertexModifierIr(ir, {}, error);
  if (!admitted) return nullptr;
  auto program = videohelper::materialprogram::compileMaterialProgram(nullptr, &*admitted, BackendTarget::OpenGl, error);
  if (!program) return nullptr;
  return std::make_shared<const videohelper::materialprogram::MaterialProgramRecord>(std::move(*program));
}

std::unique_ptr<HarmonicMIDI::grid::Visual3DScene> scene() {
  auto result = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
  result->vertices[0].position = {1, 0, 0};
  result->vertices[1].position = {0, 2, 0};
  result->vertices[2].position = {9, 9, 9};
  result->vertexCount = 3;
  result->objects[0].id = {4};
  result->objects[0].vertexCount = 2;
  result->objectCount = 1;
  return result;
}

void inputs() {
  arbitgpu::NativeFixtureSceneRuntimeInputs in;
  in.timeSeconds = 2.0f;
  in.vertexSpectrum[1] = 0.25f;
  in.vertexControls = {3.0f};
  check(cpu::priorInputs(in).timeSeconds == 2.0 && cpu::priorInputs(in).controls == in.vertexControls,
        "without a prior sample the prior inputs are this frame's");
  in.previousMotion = arbitgpu::sceneMotionSample(in);
  in.previousMotion->timeSeconds = 1.5f;
  in.previousMotion->vertexSpectrum[1] = 0.75f;
  in.previousMotion->vertexControls = {1.0f};
  const auto prior = cpu::priorInputs(in);
  check(prior.timeSeconds == 1.5 && prior.spectrum[1] == 0.75f && prior.controls == std::vector<float>({1.0f}),
        "the prior inputs are the prior sample's time, spectrum and control bank");
  in.previousMotion->vertexControls = {1.0f, 2.0f};
  check(cpu::priorInputs(in).controls == in.vertexControls,
        "a prior bank of another program reads as this frame's, as the raster's row 1 does");
}

void positions() {
  std::string error;
  const auto program = compiled(error);
  check(program != nullptr, ("the control program compiles: " + error).c_str());
  if (!program) return;
  arbitgpu::NativeFixtureSurfaceMaterialProgram root;
  root.object = {4};
  root.vertexProgram = program;
  const auto source = scene();
  arbitgpu::NativeFixtureSceneRuntimeInputs in;
  in.vertexControls = {3.0f};

  auto current = std::make_unique<HarmonicMIDI::grid::Visual3DScene>(*source);
  check(cpu::displaceScene(*current, root, cpu::currentInputs(in), error) && near(current->vertices[0].position, 4, 3, 3)
            && near(current->vertices[2].position, 9, 9, 9),
        "this frame moves the program's object by this frame's bank");

  std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>> prior;
  check(cpu::priorDisplacedPositions(*source, root, in, prior, error) && prior && prior->size() == 3
            && near((*prior)[0], 4, 3, 3),
        "without a prior sample the prior positions are this frame's, so nothing moves");

  in.previousMotion = arbitgpu::sceneMotionSample(in);
  in.previousMotion->vertexControls = {1.0f};
  check(cpu::priorDisplacedPositions(*source, root, in, prior, error) && near((*prior)[0], 2, 1, 1)
            && near((*prior)[1], 1, 3, 1) && near((*prior)[2], 9, 9, 9),
        "the prior positions read the prior control bank");

  in.previousMotion->scenePositions = std::make_shared<const std::vector<HarmonicMIDI::grid::SceneVec3>>(
      std::vector<HarmonicMIDI::grid::SceneVec3>{{0, 0, 0}, {0, 1, 0}});
  check(cpu::priorDisplacedPositions(*source, root, in, prior, error) && near((*prior)[0], 1, 1, 1)
            && near((*prior)[1], 1, 2, 1) && near((*prior)[2], 9, 9, 9),
        "the program runs on the prior frame's scene positions");

  in.previousMotion->scenePositions = std::make_shared<const std::vector<HarmonicMIDI::grid::SceneVec3>>(4);
  check(cpu::priorDisplacedPositions(*source, root, in, prior, error) && near((*prior)[0], 2, 1, 1),
        "prior positions of another scene size keep this frame's positions");

  in.previousMotion->vertexControls = {};
  in.vertexControls = {};
  check(!cpu::priorDisplacedPositions(*source, root, in, prior, error) && error.find("slot 0") != std::string::npos,
        "a bank without the program's slot is refused by name");
}
} // namespace

int main() {
  inputs();
  positions();
  if (failures != 0) return 1;
  std::puts("vertex program motion tests passed");
  return 0;
}
