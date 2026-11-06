#include "gpu_backend/fixture_vertex_modifier_shader.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// GPU vertex programs gain the ops the Ocean needs: sine, cosine and Ocean
// Waves, which loops over a wave table built from the admitted settings: an
// OpenGL or Metal texture whose rows change without a new program.
// The table and the loop's sum must be the CPU Ocean's waves and displacement.
namespace {
using videowire::VertexModifierOperation;
using videowire::VertexModifierValueType;
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

// Position and time feed Ocean Waves, whose displacement is bounded onto the position.
videowire::VertexModifierIr oceanProgram(double choppiness) {
  videowire::VertexModifierIr ir;
  ir.rootId = 4;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(2, VertexModifierOperation::timeSeconds, {}, {}),
                record(3, VertexModifierOperation::oceanWaves, {1, 2}, {9, 30, 0.5, 40, 7, choppiness}),
                record(4, VertexModifierOperation::boundedDisplacementOutput, {1, 3}, {50})};
  return ir;
}

std::string shaderFor(const videowire::VertexModifierIr& ir, BackendTarget target, std::string& error) {
  const auto admitted = videohelper::vertexmodifier::admitVertexModifierIr(ir, {}, error);
  if (!admitted) return {};
  const auto program = videohelper::materialprogram::compileMaterialProgram(nullptr, &*admitted, target, error);
  if (!program) return {};
  // The Metal wave table joins the vertex arguments after the uniforms.
  return arbitgpu::fixtureVertexModifierShader(
      "vertex VertexOut _main(VertexIn in [[stage_in]],\n constant Uniforms& u [[buffer(0)]],\n uint id) {\n"
      "// ARBIT_VERTEX_MODIFIER\n}\n",
      &*program, target, error);
}

void contract() {
  for (const auto operation : {VertexModifierOperation::scalarSin, VertexModifierOperation::scalarCos,
                               VertexModifierOperation::oceanWaves}) {
    const auto token = std::string(videowire::vertexModifierOperationWireToken(operation));
    check(videowire::vertexModifierOperationFromWireToken(token) == operation, "the new op tokens round-trip");
    check(videowire::vertexModifierOperationSchema(operation).valid, "the new ops have typed schemas");
  }
  std::string error;
  auto refused = oceanProgram(1);
  refused.records[2].parameters[3] = 0.25;
  check(!videohelper::vertexmodifier::admitVertexModifierIr(refused, {}, error)
            && error.find("Ocean") != std::string::npos,
        "a size below the smallest wave is refused by name");
  refused = oceanProgram(11);
  check(!videohelper::vertexmodifier::admitVertexModifierIr(refused, {}, error), "choppiness above 10 is refused");
}

void trig() {
  // Displacement (sin t, cos t, 0) moves the position around a unit circle.
  videowire::VertexModifierIr ir;
  ir.rootId = 7;
  ir.records = {record(1, VertexModifierOperation::importedPosition, {}, {}),
                record(2, VertexModifierOperation::timeSeconds, {}, {}),
                record(3, VertexModifierOperation::scalarSin, {2}, {}),
                record(4, VertexModifierOperation::scalarCos, {2}, {}),
                record(5, VertexModifierOperation::scalarConstant, {}, {0}),
                record(6, VertexModifierOperation::vec3Compose, {3, 4, 5}, {}),
                record(7, VertexModifierOperation::boundedDisplacementOutput, {1, 6}, {2})};
  for (const auto target : {BackendTarget::OpenGl, BackendTarget::Metal}) {
    std::string error;
    const auto shader = shaderFor(ir, target, error);
    check(shader.find("= clamp(sin(") != std::string::npos && shader.find("= clamp(cos(") != std::string::npos,
          ("sine and cosine reach the shader: " + error).c_str());
  }
}

void ocean() {
  videowire::geometry::OceanSettings settings;
  settings.windSpeed = 9; settings.headingDegrees = 30; settings.smallestWave = 0.5; settings.size = 40; settings.seed = 7;
  std::vector<videowire::geometry::WindSpectrumWave> waves;
  std::string error;
  check(videowire::geometry::oceanWaves(settings, waves, error) && !waves.empty(), "the CPU Ocean builds its waves");
  for (const auto target : {BackendTarget::OpenGl, BackendTarget::Metal}) {
    const auto shader = shaderFor(oceanProgram(1.5), target, error);
    const bool metal = target == BackendTarget::Metal;
    check(shader.find(metal ? "vertexWaves.read(uint2(0, 0))" : "texelFetch(uVertexWaves, ivec2(0, 0), 0)")
              != std::string::npos
              && shader.find("OceanWaves[") == std::string::npos,
          ("the shader reads the wave table texture, not literals: " + error).c_str());
    check(!metal || shader.find("constant Uniforms& u [[buffer(0)]],\n                       "
                                "texture2d<float, access::read> vertexWaves [[texture(0)]],") != std::string::npos,
          "the Metal vertex function takes the wave table texture after its uniforms");
    check(shader == shaderFor(oceanProgram(2.5), target, error)
              && shader.find("for (int wave = 0; wave < count; ++wave)") != std::string::npos,
          "new wave settings keep the shader text, so no recompile");
    check(shader.find(metal ? "u.vertexTime.x" : "uVertexTime.x") != std::string::npos
              && shader.find(metal ? "u.vertexTime.y" : "uVertexTime.y") != std::string::npos,
          "both the current and the prior frame evaluate the waves");
  }

  // The OpenGL table rows hold the count, the choppiness and each wave.
  {
    const auto admitted = videohelper::vertexmodifier::admitVertexModifierIr(oceanProgram(1.5), {}, error);
    const auto program = admitted ? videohelper::materialprogram::compileMaterialProgram(nullptr, &*admitted,
                                                                                         BackendTarget::OpenGl, error)
                                  : std::nullopt;
    arbitgpu::FixtureVertexWaveTable table;
    check(program && arbitgpu::fixtureVertexWaveTable(&*program, table, error) && table.rows == 1
              && table.width == 1 + 2 * waves.size() && table.texels.size() == table.width,
          ("the program builds a one-row wave table: " + error).c_str());
    if (table.texels.size() == table.width) {
      const auto& last = waves.back();
      check(table.texels[0][0] == static_cast<float>(waves.size()) && table.texels[0][1] == 1.5f,
            "texel 0 holds the wave count and choppiness");
      check(table.texels[table.width - 2][0] == static_cast<float>(last.kx)
                && table.texels[table.width - 2][3] == static_cast<float>(last.omega)
                && table.texels[table.width - 1][0] == static_cast<float>(last.phase),
            "each wave fills its two texels");
    }
  }

  // The loop body in double precision equals the CPU Ocean with choppiness.
  const double x = 3.25, y = -1.5, seconds = 2.75, choppiness = 1.5;
  double gx = 0, gy = 0, gz = 0;
  for (const auto& w : waves) {
    const auto phi = w.kx * x + w.ky * y - w.omega * seconds + w.phase;
    const auto c = w.amplitude * std::cos(phi) / std::hypot(w.kx, w.ky) * choppiness;
    gx += w.kx * c; gy += w.ky * c; gz += w.amplitude * std::sin(phi);
  }
  const auto cpu = videowire::geometry::oceanDisplacement(waves, x, y, seconds);
  check(std::abs(gx - cpu[0] * choppiness) < 1e-9 && std::abs(gy - cpu[1] * choppiness) < 1e-9
            && std::abs(gz - cpu[2]) < 1e-9,
        "the shader's sum is the CPU Ocean's displacement");
}
} // namespace

int main() {
  contract();
  trig();
  ocean();
  if (failures != 0) return 1;
  std::puts("vertex ocean waves tests passed");
  return 0;
}
