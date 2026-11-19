// The open sea moved on the GPU against the CPU replay it stands in for. The
// Lighthouse and Ocean sea (Grid, Wind Spectrum, Shade Smooth) is lowered and
// replayed on the CPU at two times. The earlier frame fills a vertex buffer behind
// a few unrelated vertices, the surface wave pass (kSurfaceWaveComputeSource, the
// backend's own source) moves it to the later time, and every water vertex is
// compared with the later CPU frame: height, normal angle, and every other field
// bit for bit. Prints the numbers and fails past the tolerances below.
// Needs a GL 4.3 context. Not in CMake; build it alone from the repository root:
//   c++ -std=c++17 -O2 -DARBIT_HAVE_VIEWPORT=1 -Ivideo-helper/src -Ishared -Iplugin/Source
//       video-helper/tests/surface_wave_gpu_test.cpp video-helper/src/gl_loader.cpp
//       -lglfw -lGL -o surface-wave-gpu-test
//   ./surface-wave-gpu-test [side=1024]
#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/GeometrySurfaceWave.h"
#include "../../shared/GeometryTimelineSceneBinding.h"
#include "../src/gl_loader.h"
#include "../src/gpu_backend/surface_wave_compute_source.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace videowire::geometry;
using HarmonicMIDI::grid::SceneVertex;
using Clock = std::chrono::steady_clock;

// A sea about 3.5 m high: a millimetre of height and a tenth of a degree of normal.
constexpr double kHeightTolerance = 1.0e-3;
constexpr double kNormalToleranceDegrees = 0.1;
constexpr std::uint32_t kLeadingVertices = 5;

double msSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int fail(const char* what, const std::string& detail = {}) {
  std::fprintf(stderr, "FAIL %s%s%s\n", what, detail.empty() ? "" : ": ", detail.c_str());
  return 1;
}

unsigned compile(arbitgl::GlFuncs& gl, std::string& log) {
  const auto shader = gl.CreateShader(GL_COMPUTE_SHADER);
  const char* source = arbitgpu::kSurfaceWaveComputeSource;
  gl.ShaderSource(shader, 1, &source, nullptr);
  gl.CompileShader(shader);
  int ok = 0;
  gl.GetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  char text[4096] {};
  if (ok != GL_TRUE) { gl.GetShaderInfoLog(shader, sizeof text, nullptr, text); log = text; return 0; }
  const auto program = gl.CreateProgram();
  gl.AttachShader(program, shader);
  gl.LinkProgram(program);
  gl.DeleteShader(shader);
  gl.GetProgramiv(program, GL_LINK_STATUS, &ok);
  if (ok != GL_TRUE) { gl.GetProgramInfoLog(program, sizeof text, nullptr, text); log = text; return 0; }
  return program;
}
} // namespace

int main(int argc, char** argv) {
  const std::uint32_t side = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 1024;
  std::string error;
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxAttributes = kMaximumAttributes;
  const auto grid = lowerGrid(7, 7, 1, side, side, 0.25f, contract, error);
  if (!grid) return fail("grid", error);
  const auto wave = lowerWindSpectrumSurfaceWave(*grid, 9, (StableId{1} << 40) + 9 * 16, 3.5, 1.5, 110, contract, error);
  if (!wave) return fail("wave", error);
  const auto shaded = lowerShadeEdit(*wave, 11, 1.0f, 30.0f, {}, error);
  if (!shaded) return fail("shade", error);
  const auto admittedContract = withAttributeContract(contract, *shaded);
  const auto admitted = admitValue(*shaded, admittedContract, error);
  if (!admitted) return fail("admit", error);

  using namespace HarmonicMIDI::grid;
  auto scene = std::make_shared<Visual3DScene>();
  scene->id = Scene3DId{301};
  scene->cameras[0].id = SceneCameraId{401};
  scene->cameraCount = 1;
  scene->activeCamera = scene->cameras[0].id;
  SceneObjectRecord object;
  object.id = SceneObjectId{101};
  object.material = SceneMaterialId{201};
  SceneMaterialRecord material;
  material.id = object.material;
  if (!appendGeometryToScene(*admitted, object, material, *scene, error)) return fail("append", error);
  TimelineSceneBinding binding;
  binding.object = object.id;
  binding.loweredPlan = encodeLoweredPlanText(lowerRuntimePlan(admittedContract, *admitted, {}, {}, true));
  const auto decoded = decodeTimelineScenePlan(binding, error);
  if (!decoded) return fail("decode", error);

  // The helper recognizes the decoded plan, as the Preview does.
  const auto program = recognizeSurfaceWaveProgram(decoded->descriptor());
  if (!program) return fail("the open-sea plan was not recognized");
  if (program->columns != side || program->rows != side || program->waves.size() != 96)
    return fail("recognized grid or wave count", std::to_string(program->columns) + "x" + std::to_string(program->rows)
        + ", " + std::to_string(program->waves.size()) + " waves");
  // One changed operation is not the open sea.
  {
    auto changed = decoded->descriptor();
    for (auto& op : changed.operations)
      if (op.code == OperationCode::shadeMesh) op.parameters[0] = kShadeFlat;
    if (recognizeSurfaceWaveProgram(changed)) return fail("a flat-shaded sea was recognized");
  }

  const double before = 1.0, after = 1.0 + 7.0 / 30.0;
  if (!replaceTimelineSceneMesh(binding, *decoded, before, *scene, error)) return fail("replay before", error);
  const std::vector<SceneVertex> start(scene->vertices.begin(), scene->vertices.begin() + scene->vertexCount);
  auto replayStart = Clock::now();
  if (!replaceTimelineSceneMesh(binding, *decoded, after, *scene, error, nullptr, true)) return fail("replay after", error);
  const double replayMs = msSince(replayStart);
  const std::vector<SceneVertex> expected(scene->vertices.begin(), scene->vertices.begin() + scene->vertexCount);
  const std::size_t count = std::size_t(side) * side;
  if (expected.size() != count) return fail("the shaded sea split vertices");

  if (!glfwInit()) return fail("glfwInit");
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  auto* window = glfwCreateWindow(1, 1, "surface-wave-gpu-test", nullptr, nullptr);
  if (!window) return fail("GL 4.3 context unavailable");
  glfwMakeContextCurrent(window);
  arbitgl::GlFuncs gl;
  if (!arbitgl::loadGlFunctions(gl, error) || !arbitgl::loadGl43Functions(gl)) return fail("GL 4.3 functions", error);
  std::string log;
  const auto shaderProgram = compile(gl, log);
  if (!shaderProgram) return fail("surface wave shader", log);

  // Leading vertices the pass must not touch, then the water at the earlier time.
  std::vector<SceneVertex> buffer(kLeadingVertices);
  for (std::uint32_t i = 0; i < kLeadingVertices; ++i)
    buffer[i].position = {float(i) + 0.5f, -float(i), 2.0f};
  buffer.insert(buffer.end(), start.begin(), start.end());
  unsigned vertices = 0, table = 0;
  gl.GenBuffers(1, &vertices);
  gl.GenBuffers(1, &table);
  gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, vertices);
  gl.BufferData(GL_SHADER_STORAGE_BUFFER, std::ptrdiff_t(buffer.size() * sizeof(SceneVertex)), buffer.data(), GL_DYNAMIC_DRAW);
  const auto rows = surfaceWaveTable(*program, after);
  gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, table);
  gl.BufferData(GL_SHADER_STORAGE_BUFFER, std::ptrdiff_t(rows.size() * sizeof(rows[0])), rows.data(), GL_STREAM_DRAW);
  gl.UseProgram(shaderProgram);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertices);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, table);
  const auto uniform = [&](const char* name, unsigned value) {
    gl.Uniform1ui(gl.GetUniformLocation(shaderProgram, name), value);
  };
  uniform("uFirst", kLeadingVertices);
  uniform("uColumns", side);
  uniform("uRows", side);
  uniform("uWaves", unsigned(rows.size()));
  // Small chunks here, so the multi-dispatch path the backend takes past 4M vertices runs too.
  const std::size_t chunk = 64 * 4096;
  glFinish();
  double gpuMs = 0;
  for (int pass = 0; pass < 3; ++pass) {
    const auto dispatchStart = Clock::now();
    for (unsigned mode = 0; mode < 2; ++mode) {
      uniform("uMode", mode);
      for (std::size_t base = 0; base < count; base += chunk) {
        uniform("uBase", unsigned(base));
        gl.DispatchCompute(unsigned((std::min(chunk, count - base) + 63) / 64), 1, 1);
      }
      gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    }
    glFinish();
    // The pass is idempotent at one time: the heights read only rest X and Y.
    gpuMs = msSince(dispatchStart);
  }
  if (glGetError() != GL_NO_ERROR) return fail("GL error after the dispatch");
  gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, vertices);
  const auto* mapped = static_cast<const SceneVertex*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
      std::ptrdiff_t(buffer.size() * sizeof(SceneVertex)), GL_MAP_READ_BIT));
  if (!mapped) return fail("map the vertex buffer");
  std::vector<SceneVertex> moved(mapped, mapped + buffer.size());
  gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER);

  int failures = 0;
  for (std::uint32_t i = 0; i < kLeadingVertices; ++i)
    if (std::memcmp(&moved[i], &buffer[i], sizeof(SceneVertex)) != 0) ++failures;
  if (failures) std::fprintf(stderr, "FAIL %d leading vertices changed\n", failures);
  double maxHeight = 0, sumSquares = 0, maxAngle = 0, sumAngle = 0, maxExpectedHeight = 0, maxMotion = 0;
  std::size_t otherFields = 0, worst = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& gpu = moved[kLeadingVertices + i];
    const auto& cpu = expected[i];
    const double dz = std::abs(double(gpu.position.z) - cpu.position.z);
    if (dz > maxHeight) { maxHeight = dz; worst = i; }
    sumSquares += dz * dz;
    maxExpectedHeight = std::max(maxExpectedHeight, std::abs(double(cpu.position.z)));
    maxMotion = std::max(maxMotion, std::abs(double(cpu.position.z) - start[i].position.z));
    const double dot = double(gpu.normal.x) * cpu.normal.x + double(gpu.normal.y) * cpu.normal.y
        + double(gpu.normal.z) * cpu.normal.z;
    const double lengths = std::sqrt((double(gpu.normal.x) * gpu.normal.x + double(gpu.normal.y) * gpu.normal.y
        + double(gpu.normal.z) * gpu.normal.z) * (double(cpu.normal.x) * cpu.normal.x
        + double(cpu.normal.y) * cpu.normal.y + double(cpu.normal.z) * cpu.normal.z));
    const double angle = std::acos(std::clamp(dot / lengths, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
    maxAngle = std::max(maxAngle, angle);
    sumAngle += angle;
    if (std::memcmp(&gpu.position.x, &cpu.position.x, sizeof(float)) != 0
        || std::memcmp(&gpu.position.y, &cpu.position.y, sizeof(float)) != 0
        || std::memcmp(&gpu.tangent, &cpu.tangent, sizeof cpu.tangent) != 0
        || std::memcmp(&gpu.uv, &cpu.uv, sizeof cpu.uv) != 0
        || std::memcmp(&gpu.color, &cpu.color, sizeof cpu.color) != 0
        || std::memcmp(&gpu.roughness, &cpu.roughness, sizeof cpu.roughness) != 0)
      ++otherFields;
  }
  std::printf("sea %ux%u, %zu waves, %.4f s apart: largest height %.3f m, largest move %.3f m\n", side, side,
      rows.size(), after - before, maxExpectedHeight, maxMotion);
  std::printf("GPU vs CPU height: max %.3g m (vertex %zu), rms %.3g m\n", maxHeight, worst,
      std::sqrt(sumSquares / double(count)));
  std::printf("GPU vs CPU normal: max %.3g deg, mean %.3g deg\n", maxAngle, sumAngle / double(count));
  std::printf("other fields differing: %zu of %zu\n", otherFields, count);
  std::printf("CPU replay %.1f ms, GPU heights+normals %.2f ms (glFinish wall)\n", replayMs, gpuMs);
  if (maxHeight > kHeightTolerance) ++failures, std::fprintf(stderr, "FAIL height past %.3g m\n", kHeightTolerance);
  if (maxAngle > kNormalToleranceDegrees)
    ++failures, std::fprintf(stderr, "FAIL normal past %.3g deg\n", kNormalToleranceDegrees);
  if (otherFields) ++failures, std::fprintf(stderr, "FAIL fields other than Z and the normal changed\n");
  glfwDestroyWindow(window);
  glfwTerminate();
  std::printf(failures ? "FAILED\n" : "PASSED\n");
  return failures ? 1 : 0;
}
