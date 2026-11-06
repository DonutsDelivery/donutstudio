#include "gpu_backend/fixture_vertex_modifier_shader.h"
#include "vertex_program_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Vertex Control Parameter reads the control bank on every engine. The bank
// takes each record's authored value, a runtime value replaces it, two records
// on one slot must agree, and a record from before the value existed reads 0.
// Raster's GL and Metal text read the slot from the control texture without the
// value in the source, the prior-frame evaluation reads the prior bank's row for
// motion vectors, and Path Trace's host displacement reads the same bank.
namespace {
using videowire::VertexModifierOperation;
using videohelper::materialprogram::BackendTarget;
namespace control = videowire::vertexcontrol;

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

// Position plus (c, c, c), where c reads control slot 5.
videowire::VertexModifierIr program(std::initializer_list<double> controlParameters) {
  videowire::VertexModifierIr ir;
  ir.rootId = 3;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(2, VertexModifierOperation::controlParameter, {}, controlParameters),
                record(4, VertexModifierOperation::vec3Compose, {2, 2, 2}, {}),
                record(3, VertexModifierOperation::boundedDisplacementOutput, {1, 4}, {10})};
  return ir;
}

std::optional<videohelper::materialprogram::MaterialProgramRecord> compile(const videowire::VertexModifierIr& ir,
                                                                          BackendTarget target, std::string& error) {
  const auto admitted = videohelper::vertexmodifier::admitVertexModifierIr(ir, {}, error);
  if (!admitted) return std::nullopt;
  return videohelper::materialprogram::compileMaterialProgram(nullptr, &*admitted, target, error);
}

void bank() {
  std::vector<float> values;
  std::string error;
  check(control::bank(program({5, 0.75}), nullptr, values, error) && values.size() == 6 && values[5] == 0.75f
            && values[0] == 0.0f,
        "the bank holds the authored value at its slot");
  const std::map<std::string, double> runtime {{control::runtimeKey(2), -2.5}};
  check(control::runtimeKey(2) == "visual2/value", "the runtime key names the record and its value");
  check(control::bank(program({5, 0.75}), &runtime, values, error) && values[5] == -2.5f,
        "a runtime value replaces the authored value");
  check(control::bank(program({5}), nullptr, values, error) && values.size() == 6 && values[5] == 0.0f,
        "a record without a value reads 0");
  check(control::bank(program({5, 2e7}), nullptr, values, error) && values[5] == 1e6f,
        "a value clamps to the admitted range");

  auto two = program({5, 1.0});
  two.records.push_back(record(9, VertexModifierOperation::controlParameter, {}, {5, 1.0}));
  check(control::bank(two, nullptr, values, error), "two records that agree share a slot");
  two.records.back().parameters[1] = 2.0;
  check(!control::bank(two, nullptr, values, error) && error.find("slot 5") != std::string::npos,
        "two records that disagree on a slot are refused by name");
  const std::map<std::string, double> bad {{control::runtimeKey(2), std::nan("")}};
  check(!control::bank(program({5, 0.0}), &bad, values, error) && error.find("not finite") != std::string::npos,
        "a non-finite runtime value is refused");

  const auto texels = control::texels({1, 2, 3, 4, 5, 6});
  check(texels.size() == 8 && texels[5] == 6.0f && texels[7] == 0.0f, "texels pad the bank to whole RGBA texels");
}

void motion() {
  const std::vector<float> current {1, 2, 3, 4, 5};
  const std::vector<float> prior {6, 7, 8, 9, 10};
  const auto rows = control::motionTexels(current, &prior);
  check(rows.size() == 16 && rows[4] == 5.0f && rows[7] == 0.0f, "row 0 holds this frame's bank");
  check(rows[8] == 6.0f && rows[12] == 10.0f && rows[15] == 0.0f, "row 1 holds the prior frame's bank");
  const auto still = control::motionTexels(current, nullptr);
  check(still.size() == 16 && std::equal(still.begin(), still.begin() + 8, still.begin() + 8),
        "without a prior frame both rows match, so the vertex has no motion");
  const std::vector<float> other {1, 2};
  const auto changed = control::motionTexels(current, &other);
  check(std::equal(changed.begin(), changed.begin() + 8, changed.begin() + 8),
        "a prior bank of another program repeats this frame's row");
}

void admission() {
  std::string error;
  check(videohelper::vertexmodifier::admitVertexModifierIr(program({5, 0.75}), {}, error).has_value(),
        ("a record with a slot and value admits: " + error).c_str());
  check(videohelper::vertexmodifier::admitVertexModifierIr(program({5}), {}, error).has_value(),
        ("a record with only a slot still admits: " + error).c_str());
  check(!videohelper::vertexmodifier::admitVertexModifierIr(program({5, 0.75, 1.0}), {}, error).has_value(),
        "a record with three parameters is refused");
}

void shaders() {
  for (const auto target : {BackendTarget::OpenGl, BackendTarget::Metal}) {
    const bool metal = target == BackendTarget::Metal;
    std::string error;
    const auto low = compile(program({5, 0.25}), target, error);
    const auto high = compile(program({5, 9.0}), target, error);
    check(low && high, ("the control program compiles: " + error).c_str());
    if (!low || !high) continue;
    check(low->resources().controlParameterSlots == 6, "the program declares six control slots");
    const std::string source = "vertex VertexOut _main(VertexIn in [[stage_in]],\n"
                               " constant Uniforms& u [[buffer(0)]],\n uint id) {\n// ARBIT_VERTEX_MODIFIER\n}\n";
    const auto shader = arbitgpu::fixtureVertexModifierShader(source.c_str(), &*low, target, error);
    check(!shader.empty(), ("Raster writes the control program: " + error).c_str());
    check(shader.find(metal ? "vertexControls.read(uint2(1, 0))[1]" : "texelFetch(uVertexControls, ivec2(1, 0), 0)[1]")
              != std::string::npos,
          metal ? "Metal reads slot 5 from texel 1, lane 1" : "OpenGL reads slot 5 from texel 1, lane 1");
    check(shader.find(metal ? "vertexControls.read(uint2(1, 1))[1]" : "texelFetch(uVertexControls, ivec2(1, 1), 0)[1]")
              != std::string::npos,
          metal ? "Metal's prior-frame evaluation reads slot 5 from row 1"
                : "OpenGL's prior-frame evaluation reads slot 5 from row 1");
    check(shader == arbitgpu::fixtureVertexModifierShader(source.c_str(), &*high, target, error),
          "the authored value never enters the shader text");
    if (metal)
      check(shader.find("texture2d<float, access::read> vertexControls [[texture(1)]],") != std::string::npos,
            "Metal declares the control texture argument");
  }
}

void hostDisplacement() {
  std::string error;
  const auto compiled = compile(program({5, 0.0}), BackendTarget::OpenGl, error);
  check(bool(compiled), ("the control program compiles: " + error).c_str());
  if (!compiled) return;
  auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
  scene->vertices[0].position = {0.3f, -1.2f, 2.5f};
  scene->vertexCount = 1;
  scene->objects[0].id = {3};
  scene->objects[0].vertexCount = 1;
  scene->objectCount = 1;
  videohelper::vertexprogramcpu::Inputs inputs;
  check(!videohelper::vertexprogramcpu::displaceObject(*scene, {3}, *compiled, inputs, error)
            && error.find("slot 5") != std::string::npos,
        "Path Trace names a control slot the bank lacks");
  inputs.controls = {0, 0, 0, 0, 0, 1.5f};
  check(videohelper::vertexprogramcpu::displaceObject(*scene, {3}, *compiled, inputs, error),
        ("Path Trace reads the control bank: " + error).c_str());
  const auto& moved = scene->vertices[0].position;
  check(std::abs(moved.x - 1.8f) < 1e-6f && std::abs(moved.y - 0.3f) < 1e-6f && std::abs(moved.z - 4.0f) < 1e-6f,
        "the host moves the vertex by the control value");
}
} // namespace

int main() {
  bank();
  motion();
  admission();
  shaders();
  hostDisplacement();
  if (failures != 0) return 1;
  std::puts("vertex control bank tests passed");
  return 0;
}
