// Per-frame CPU cost of the Lighthouse and Ocean sea: a 1024 by 1024 grid under the
// wind spectrum, lowered the way the starter lowers it (Grid, Wind Spectrum, Shade
// Smooth), then admitted, decoded and replaced for consecutive timeline frames the
// way the helper does. Prints each phase's time. Not a pass/fail test: it exits 0
// unless a phase refuses, and the last line hashes the mesh so two builds compare.
// Not in CMake; build it alone:
//   c++ -std=c++17 -O3 -DNDEBUG -Ivideo-helper/src -Ishared -Iplugin/Source
//       video-helper/tests/timeline_sea_frame_cost_bench.cpp -o sea-bench
//   ./sea-bench [side=1024] [frames=4]
#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/GeometryTimelineSceneBinding.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {
using namespace videowire::geometry;
using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

PortContract seaContract() {
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxAttributes = kMaximumAttributes;
  return contract;
}

int fail(const char* phase, const std::string& error) {
  std::fprintf(stderr, "FAIL %s: %s\n", phase, error.c_str());
  return 1;
}
} // namespace

int main(int argc, char** argv) {
  const std::uint32_t side = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 1024;
  const int frames = argc > 2 ? std::atoi(argv[2]) : 4;
  std::string error;
  const auto contract = seaContract();
  auto start = Clock::now();
  const auto grid = lowerGrid(7, 7, 1, side, side, 0.25f, contract, error);
  if (!grid) return fail("grid", error);
  const auto wave = lowerWindSpectrumSurfaceWave(*grid, 9, (StableId{1} << 40) + 9 * 16,
      3.5, 1.5, 110, contract, error);
  if (!wave) return fail("wave", error);
  const auto shaded = lowerShadeEdit(*wave, 11, 1.0f, 30.0f, {}, error);
  if (!shaded) return fail("shade", error);
  std::printf("lower %.1f ms\n", msSince(start));

  start = Clock::now();
  const auto admittedContract = withAttributeContract(contract, *shaded);
  const auto admitted = admitValue(*shaded, admittedContract, error);
  if (!admitted) return fail("admit", error);
  std::printf("admit %.1f ms\n", msSince(start));

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
  start = Clock::now();
  if (!appendGeometryToScene(*admitted, object, material, *scene, error)) return fail("append", error);
  std::printf("append %.1f ms (%zu vertices, %zu indices)\n", msSince(start), scene->vertexCount, scene->indexCount);

  {
    const auto check = validateVisual3DScene(*scene);
    std::printf("appended scene validation code %d record %zu\n", static_cast<int>(check.code), check.recordIndex);
  }
  TimelineSceneBinding binding;
  binding.object = object.id;
  start = Clock::now();
  binding.loweredPlan = encodeLoweredPlanText(lowerRuntimePlan(admittedContract, *admitted, {}, {}, true));
  std::printf("encode %.1f ms (%zu plan bytes)\n", msSince(start), binding.loweredPlan.size());

  start = Clock::now();
  const auto decoded = decodeTimelineScenePlan(binding, error);
  if (!decoded) return fail("decode", error);
  std::printf("decode %.1f ms\n", msSince(start));

  start = Clock::now();
  if (!validTimelineSceneBinding(binding, *scene, *decoded, error)) return fail("binding", error);
  std::printf("validate binding %.1f ms\n", msSince(start));

  for (int frame = 0; frame < frames; ++frame) {
    const double seconds = 1.0 + frame / 30.0;
    start = Clock::now();
    if (!replaceTimelineSceneMesh(binding, *decoded, seconds, *scene, error, nullptr, frame > 0))
      return fail("replace", error);
    std::printf("frame %d replace%s %.1f ms\n", frame, frame > 0 ? " (verified)" : "", msSince(start));
  }
  start = Clock::now();
  const bool valid = validateVisual3DScene(*scene).valid();
  std::printf("validateVisual3DScene %.1f ms (%s)\n", msSince(start), valid ? "valid" : "invalid");
  // An FNV-1a hash of the last frame's vertex and index bytes, so a before/after pair
  // can show the same mesh bit for bit.
  std::uint64_t hash = 1469598103934665603ull;
  const auto mix = [&](const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
  };
  for (std::size_t i = 0; i < scene->vertexCount; ++i) {
    const auto& v = scene->vertices[i];
    mix(&v.position, sizeof v.position); mix(&v.normal, sizeof v.normal); mix(&v.uv, sizeof v.uv);
  }
  for (std::size_t i = 0; i < scene->indexCount; ++i) mix(&scene->indices[i], sizeof scene->indices[i]);
  std::printf("mesh hash %016llx\n", static_cast<unsigned long long>(hash));
  // SEA_DUMP=path writes the last frame's grid vertices, side * side of them in grid
  // order, as "side\n" then six floats each: position, then normal.
  if (const char* path = std::getenv("SEA_DUMP")) {
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) return fail("dump", path);
    std::fprintf(file, "%u\n", side);
    for (std::size_t i = 0; i < std::size_t(side) * side && i < scene->vertexCount; ++i) {
      const auto& v = scene->vertices[i];
      const float values[6] {v.position.x, v.position.y, v.position.z, v.normal.x, v.normal.y, v.normal.z};
      std::fwrite(values, sizeof values, 1, file);
    }
    std::fclose(file);
  }
  return 0;
}
