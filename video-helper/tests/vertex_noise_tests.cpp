#include "gpu_backend/fixture_vertex_modifier_shader.h"
#include "vertex_program_cpu.h"

#include <cmath>
#include <cstdio>
#include <string>

// Vertex Noise 3D runs on every engine: the shared value noise stays in
// [-1, 1], is continuous and repeats for a seed, Raster's GL and Metal text
// spells out its hash, and Path Trace's host displacement moves a vertex by the
// same noise.
namespace {
using videowire::VertexModifierOperation;
using videohelper::materialprogram::BackendTarget;
namespace noise = videowire::vertexnoise;

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

videowire::VertexModifierIr program(VertexModifierOperation source) {
  videowire::VertexModifierIr ir;
  ir.rootId = 3;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(2, source, {1}, {1.5, 42}),
                record(3, VertexModifierOperation::boundedDisplacementOutput, {1, 2}, {10})};
  if (source != VertexModifierOperation::vec3Noise3d) {
    ir.records.insert(ir.records.begin() + 2, record(4, VertexModifierOperation::vec3Compose, {2, 2, 2}, {}));
    ir.records.back().inputs[1] = 4;
  }
  return ir;
}

std::optional<videohelper::materialprogram::MaterialProgramRecord> compile(const videowire::VertexModifierIr& ir,
                                                                          BackendTarget target, std::string& error) {
  const auto admitted = videohelper::vertexmodifier::admitVertexModifierIr(ir, {}, error);
  if (!admitted) return std::nullopt;
  return videohelper::materialprogram::compileMaterialProgram(nullptr, &*admitted, target, error);
}

void values() {
  bool inRange = true, continuous = true;
  for (int i = 0; i < 2000; ++i) {
    const float x = -7.3f + 0.0137f * i, y = 0.21f * std::sin(0.01f * i), z = 3.0f - 0.004f * i;
    const auto v = noise::noise(x, y, z, 1.7f, 9);
    inRange = inRange && std::isfinite(v) && v >= -1.0f && v <= 1.0f;
    continuous = continuous && std::abs(noise::noise(x + 1e-4f, y, z, 1.7f, 9) - v) < 0.01f;
  }
  check(inRange, "noise stays in [-1, 1]");
  check(continuous, "noise is continuous");
  check(noise::noise(0.4f, 1.1f, -2.6f, 1, 5) == noise::noise(0.4f, 1.1f, -2.6f, 1, 5), "a seed repeats");
  check(noise::noise(0.4f, 1.1f, -2.6f, 1, 5) != noise::noise(0.4f, 1.1f, -2.6f, 1, 6), "seeds differ");
  check(noise::noise(2, -3, 4, 1, 7) == noise::corner(2, -3, 4, 7), "a lattice point is its corner value");
  check(noise::noise(1, -1.5f, 2, 2, 7) == noise::corner(2, -3, 4, 7), "frequency scales the lattice");
}

void shaders() {
  for (const auto target : {BackendTarget::OpenGl, BackendTarget::Metal}) {
    std::string error;
    const auto compiled = compile(program(VertexModifierOperation::vec3Noise3d), target, error);
    check(bool(compiled), ("the noise program compiles: " + error).c_str());
    if (!compiled) continue;
    const auto shader = arbitgpu::fixtureVertexModifierShader("// ARBIT_VERTEX_MODIFIER", &*compiled, target, error);
    check(!shader.empty(), ("Raster writes the noise program: " + error).c_str());
    check(shader.find("0x7feb352du") != std::string::npos && shader.find("43u") != std::string::npos
              && shader.find("44u") != std::string::npos,
          "the shader hashes the three lane seeds");
    check(shader.find(target == BackendTarget::Metal ? "int3 cell" : "ivec3 cell") != std::string::npos,
          "the shader uses the backend's integer vector");
  }
}

void hostDisplacement() {
  std::string error;
  const auto compiled = compile(program(VertexModifierOperation::scalarNoise3d), BackendTarget::OpenGl, error);
  check(bool(compiled), ("the scalar noise program compiles: " + error).c_str());
  if (!compiled) return;
  auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
  scene->vertices[0].position = {0.3f, -1.2f, 2.5f};
  scene->vertexCount = 1;
  scene->objects[0].id = {3};
  scene->objects[0].vertexCount = 1;
  scene->objectCount = 1;
  check(videohelper::vertexprogramcpu::displaceObject(*scene, {3}, *compiled, {}, error),
        ("Path Trace runs vertex noise: " + error).c_str());
  const auto n = noise::noise(0.3f, -1.2f, 2.5f, 1.5f, 42);
  const auto& moved = scene->vertices[0].position;
  check(std::abs(moved.x - (0.3f + n)) < 1e-6f && std::abs(moved.y - (-1.2f + n)) < 1e-6f
            && std::abs(moved.z - (2.5f + n)) < 1e-6f,
        "the host moves the vertex by the shared noise");
}

} // namespace

int main() {
  values();
  shaders();
  hostDisplacement();
  if (failures != 0) return 1;
  std::puts("vertex noise tests passed");
  return 0;
}
