#include "vertex_program_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

// The host displacement Path Trace runs before its BVH: a compiled vertex
// program moves one object's vertices as the raster shader would, the Ocean
// Waves op matches the CPU Ocean, and vertex noise runs on the host too.
namespace {
using videowire::VertexModifierOperation;
using videohelper::materialprogram::BackendTarget;

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

std::optional<videohelper::materialprogram::MaterialProgramRecord> compile(const videowire::VertexModifierIr& ir,
                                                                          std::string& error) {
  const auto admitted = videohelper::vertexmodifier::admitVertexModifierIr(ir, {}, error);
  if (!admitted) return std::nullopt;
  return videohelper::materialprogram::compileMaterialProgram(nullptr, &*admitted, BackendTarget::OpenGl, error);
}

std::unique_ptr<HarmonicMIDI::grid::Visual3DScene> sceneOf(const std::vector<std::array<float, 3>>& points) {
  auto scene = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
  for (std::size_t i = 0; i < points.size(); ++i)
    scene->vertices[i].position = {points[i][0], points[i][1], points[i][2]};
  scene->vertexCount = points.size();
  scene->objects[0].id = {3};
  scene->objects[0].vertexCount = static_cast<std::uint32_t>(points.size());
  scene->objectCount = 1;
  return scene;
}

void ocean() {
  std::string error;
  videowire::VertexModifierIr ir;
  ir.rootId = 4;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(2, VertexModifierOperation::timeSeconds, {}, {}),
                record(3, VertexModifierOperation::oceanWaves, {1, 2}, {9, 30, 0.5, 40, 7, 1.5}),
                record(4, VertexModifierOperation::boundedDisplacementOutput, {1, 3}, {50})};
  const auto program = compile(ir, error);
  check(bool(program), ("the Ocean program compiles: " + error).c_str());
  if (!program) return;
  const std::vector<std::array<float, 3>> rest{{0, 0, 0}, {3.25f, -1.5f, 0}, {-7, 2, 0.5f}};
  auto scene = sceneOf(rest);
  videohelper::vertexprogramcpu::Inputs inputs;
  inputs.timeSeconds = 2.75;
  check(videohelper::vertexprogramcpu::displaceObject(*scene, {3}, *program, inputs, error),
        ("the host runs the Ocean program: " + error).c_str());

  videowire::geometry::OceanSettings settings;
  settings.windSpeed = 9; settings.headingDegrees = 30; settings.smallestWave = 0.5; settings.size = 40; settings.seed = 7;
  std::vector<videowire::geometry::WindSpectrumWave> waves;
  check(videowire::geometry::oceanWaves(settings, waves, error), "the CPU Ocean builds its waves");
  double worst = 0;
  for (std::size_t i = 0; i < rest.size(); ++i) {
    const auto d = videowire::geometry::oceanDisplacement(waves, rest[i][0], rest[i][1], 2.75);
    const auto& moved = scene->vertices[i].position;
    worst = std::max({worst, std::abs(moved.x - (rest[i][0] + d[0] * 1.5)),
                      std::abs(moved.y - (rest[i][1] + d[1] * 1.5)), std::abs(moved.z - (rest[i][2] + d[2]))});
  }
  check(worst < 1e-4, "each vertex moves by the CPU Ocean displacement with choppiness");

  std::string missing;
  check(!videohelper::vertexprogramcpu::displaceObject(*scene, {9}, *program, inputs, missing),
        "a program for an absent object is refused");
}

void bounded() {
  // Displacement (sin t, cos t, 0) * 10 is clamped to the 0.5 bound, as the shader does.
  std::string error;
  videowire::VertexModifierIr ir;
  ir.rootId = 9;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(2, VertexModifierOperation::timeSeconds, {}, {}),
                record(3, VertexModifierOperation::scalarSin, {2}, {}),
                record(4, VertexModifierOperation::scalarCos, {2}, {}),
                record(5, VertexModifierOperation::scalarConstant, {}, {0}),
                record(6, VertexModifierOperation::vec3Compose, {3, 4, 5}, {}),
                record(7, VertexModifierOperation::scalarConstant, {}, {10}),
                record(8, VertexModifierOperation::vec3Scale, {6, 7}, {}),
                record(9, VertexModifierOperation::boundedDisplacementOutput, {1, 8}, {0.5})};
  const auto program = compile(ir, error);
  check(bool(program), ("the trig program compiles: " + error).c_str());
  if (!program) return;
  auto scene = sceneOf({{1, 1, 1}});
  videohelper::vertexprogramcpu::Inputs inputs;
  inputs.timeSeconds = 0.5;
  check(videohelper::vertexprogramcpu::displaceObject(*scene, {3}, *program, inputs, error), "the trig program runs");
  const auto& moved = scene->vertices[0].position;
  check(std::abs(moved.x - (1 + 0.5 * std::sin(0.5))) < 1e-5 && std::abs(moved.y - (1 + 0.5 * std::cos(0.5))) < 1e-5
            && moved.z == 1.0f,
        "sine and cosine move the vertex by the bounded displacement");
}

void noise() {
  std::string error;
  videowire::VertexModifierIr ir;
  ir.rootId = 3;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(2, VertexModifierOperation::vec3Noise3d, {1}, {1, 0}),
                record(3, VertexModifierOperation::boundedDisplacementOutput, {1, 2}, {1})};
  const auto program = compile(ir, error);
  check(bool(program), ("the noise program compiles: " + error).c_str());
  if (!program) return;
  auto scene = sceneOf({{0.25f, 0.5f, 0.75f}});
  check(videohelper::vertexprogramcpu::displaceObject(*scene, {3}, *program, {}, error),
        ("vertex noise runs on the host: " + error).c_str());
}
} // namespace

int main() {
  ocean();
  bounded();
  noise();
  if (failures != 0) return 1;
  std::puts("vertex program cpu tests passed");
  return 0;
}
